"""
spike_margin_map.py — フルシンセ識別可能性スパイク（margin 地図）
================================================================================
仕様 v5 §7 の運用規則に基づき、変調先ペアの margin を測る:

  ターゲット: エンベロープ1本を変調先 m_true に深さ d_true でアサインして合成
  掃引:       候補変調先 m_cand の深さを [-0.65, 0.65] で掃引し、最小損失を記録
  margin[t,c] = minloss[t,c] − minloss[t,t]
  縮退判定:   margin < ε（ε = 同一構成の再合成＋微小摂動によるノイズ床の95%点）

注意（結果の読み方）:
  - 掃引は深さ1自由度のみの簡易版。measured margin ≥ true margin なので、
    「margin が小さい → 確実に縮退疑い」「大きい → 深さ単独では真似できない
    （他パラメータ併用の可能性は残る）」という非対称な読みになる。
  - ラベル固定: アクティブなエンベロープは1本のみ（置換対称は発生しない）。
  - margin はプローブのエンベロープ形状に依存する（本スクリプトは1形状）。

使い方:
  python spike_margin_map.py --sr 16000
  python spike_margin_map.py --sr 16000 --dests 0,4,5,15,16 --points 27
  python spike_margin_map.py --sr 44100 --chunk 8        # 44.1kHz はチャンク小さめ
主な引数:
  --dests   対象変調先の0始まり添字（省略時 0..16 全部）
  --points  掃引点数（既定 53、範囲 [-0.65,0.65]）
  --dtrue   真の深さ（既定 0.6。掃引格子に乗る値にすること）
  --vca     on にすると VCA エンベロープ(env1→level1, 深さ0.5)を常設
            （その場合 level1 を含むセルはスキップ）
  --chunk   1回の順方向に載せる掃引点数（メモリ調整用）
"""
import argparse, csv, time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diff_synth_tf as ds
from diff_synth_fast import synth_fast

DEST_NAMES = [s[0] for s in ds.dest_specs(16000)]
K, M = ds.K_ENVS, ds.M_DEST


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--toff", type=float, default=1.0)
    ap.add_argument("--f0", type=float, default=220.0)
    ap.add_argument("--dests", type=str, default="")
    ap.add_argument("--points", type=int, default=53)
    ap.add_argument("--dtrue", type=float, default=0.6)
    ap.add_argument("--vca", choices=["on", "off"], default="off")
    ap.add_argument("--probe", choices=["env", "lfo"], default="env",
                    help="プローブソースのタイプ（v7）")
    ap.add_argument("--lfo_rate", type=float, default=3.0)
    ap.add_argument("--lfo_phase", type=float, default=0.25)
    ap.add_argument("--lfo_morph", type=float, default=0.5)
    ap.add_argument("--cutoff_base", type=float, default=0.35,
                    help="cutoff の静的ベース（正規化値）。0.65 ≈ 680Hz, 0.35 ≈ 157Hz")
    ap.add_argument("--chunk", type=int, default=14)
    ap.add_argument("--nfloor", type=int, default=16)
    ap.add_argument("--out", type=str, default="spike")
    return ap.parse_args()


def base_static(cutoff_base=0.35):
    theta = np.full((M,), 0.35, np.float32)
    theta[2], theta[3] = 0.30, 0.40          # fin 正準順序（マージン0.10）
    theta[4] = cutoff_base                   # cutoff は対数 denorm なので物理値基準で指定
    return theta

PROBE_ENV = np.array([0.030, 0.0, 0.200, 0.4, 0.150], np.float32)  # a,h,d,s,r
VCA_ENV   = np.array([0.005, 0.0, 0.100, 0.8, 0.200], np.float32)
OSC_MORPH = np.array([0.7, 0.3], np.float32)
GMK       = 0.25


def build_batch(depths, dest, args, vca_on, probe=None):
    """深さ配列 depths を候補変調先 dest に載せたバッチ構成を返す。
    probe: None なら args.probe に従う。'env'/'lfo' 指定で上書き（cross-type 用）"""
    B = len(depths)
    probe = probe or args.probe
    theta = np.tile(base_static(args.cutoff_base), (B, 1))
    src = np.tile(PROBE_ENV, (B, K, 1)).astype(np.float32)
    src[:, 1, :] = VCA_ENV
    styp = np.ones((B, K), np.float32)          # 既定は全 ENV
    if probe == "lfo":
        src[:, 0, :3] = [args.lfo_rate, args.lfo_phase, args.lfo_morph]
        src[:, 0, 3:] = 0.0
        styp[:, 0] = 0.0
    R = np.zeros((B, K, M), np.float32)
    R[:, 0, dest] = depths
    if vca_on:
        R[:, 1, 0] = 0.5
    osc = np.tile(OSC_MORPH, (B, 1))
    gmk = np.full((B,), GMK, np.float32)
    return theta, osc, gmk, src, R, styp


def main():
    args = parse_args()
    SR, T, T_OFF = args.sr, args.T, args.toff
    N = int(SR * T)
    fmax = args.f0 * 2 ** (100 / 1200)
    HS, HQ = int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2)
    f0 = tf.constant([args.f0], tf.float32)
    dests = list(range(M)) if not args.dests else [int(s) for s in args.dests.split(",")]
    sweep = np.linspace(-0.65, 0.65, args.points).astype(np.float32)
    assert np.min(np.abs(sweep - args.dtrue)) < 1e-6, "--dtrue が掃引格子に乗っていません"
    vca_on = args.vca == "on"
    if vca_on and 0 in dests:
        print("[note] --vca on のため level1 (m=0) を含むセルはスキップします")
    C = args.chunk

    @tf.function
    def fwd(theta, osc, gmk, env, R, f0b, styp):
        return synth_fast(theta, osc, gmk, env, R, f0b, T_OFF, N, SR, HS, HQ,
                          src_type=styp)

    @tf.function
    def loss_vs(x, xt):
        return ds.mrstft_loss(x, xt, fft_sizes=(256, 512, 1024, 2048))

    def synth_np(depths, dest, probe=None):
        """チャンク処理＋パディングで同一バッチ形状を維持（再トレース回避）"""
        outs = []
        for s0 in range(0, len(depths), C):
            d = depths[s0:s0 + C]
            pad = C - len(d)
            d = np.concatenate([d, np.repeat(d[-1:], pad)]) if pad else d
            t, o, g, e, R, st = build_batch(d, dest, args, vca_on, probe=probe)
            x = fwd(tf.constant(t), tf.constant(o), tf.constant(g),
                    tf.constant(e), tf.constant(R),
                    tf.tile(f0, [len(d)]), tf.constant(st)).numpy()
            outs.append(x[:len(d) - pad] if pad else x)
        return np.concatenate(outs, 0)

    def sweep_minloss(x_t, dest, probe=None):
        """候補 (dest, probe型) の深さ掃引 → (minloss, argmin, 曲線)"""
        xt_tile_l = tf.constant(np.tile(x_t, (C, 1)))
        ls = []
        for s0 in range(0, len(sweep), C):
            d = sweep[s0:s0 + C]
            pad = C - len(d)
            dp = np.concatenate([d, np.repeat(d[-1:], pad)]) if pad else d
            t, o, g, e, R, st = build_batch(dp, dest, args, vca_on, probe=probe)
            x = fwd(tf.constant(t), tf.constant(o), tf.constant(g),
                    tf.constant(e), tf.constant(R), tf.tile(f0, [C]),
                    tf.constant(st))
            l = loss_vs(x, xt_tile_l).numpy()
            ls.extend(l[:len(l) - pad] if pad else l)
        ls = np.array(ls)
        return ls.min(), sweep[ls.argmin()], ls

    fc_phys = 20.0 * (0.45 * SR / 20.0) ** args.cutoff_base
    print(f"base patch: cutoff θ={args.cutoff_base:.2f} → {fc_phys:.0f} Hz, "
          f"f0={args.f0:.0f} Hz, d_true={args.dtrue}")
    t_start = time.time()

    # ---------------- ノイズ床 ε（§7 運用規則）----------------
    rng = np.random.default_rng(0)
    ref_dest = dests[0] if not (vca_on and dests[0] == 0) else dests[1]
    x_ref = synth_np(np.array([args.dtrue], np.float32), ref_dest)
    perts = []
    for _ in range(args.nfloor):
        t, o, g, e, R, st_f = build_batch(np.array([args.dtrue], np.float32), ref_dest, args, vca_on)
        t += rng.normal(0, 5e-4, t.shape).astype(np.float32)
        o = np.clip(o + rng.normal(0, 5e-4, o.shape).astype(np.float32), 0, 1)
        g = np.clip(g + rng.normal(0, 5e-4, g.shape).astype(np.float32), 0, 1)
        e[..., [0, 1, 2, 4]] *= (1 + rng.normal(0, 1e-3, e[..., [0, 1, 2, 4]].shape)).astype(np.float32)
        e[..., 3] = np.clip(e[..., 3] + rng.normal(0, 5e-4, e[..., 3].shape).astype(np.float32), 0, 1)
        R += rng.normal(0, 5e-4, R.shape).astype(np.float32) * (np.abs(R) > 0)
        perts.append((t, o, g, e, R))  # src_type は共通 st_f を使用
    losses_f = []
    for s0 in range(0, args.nfloor, C):
        batch = perts[s0:s0 + C]
        pad = C - len(batch)
        batch = batch + [batch[-1]] * pad
        stack = [np.concatenate([b[i] for b in batch], 0) for i in range(5)]
        st_b = np.tile(st_f, (C, 1)).astype(np.float32)
        x = fwd(*[tf.constant(v) for v in stack], tf.tile(f0, [C]), tf.constant(st_b))
        l = loss_vs(x, tf.tile(tf.constant(x_ref), [C, 1])).numpy()
        losses_f.extend(l[:len(l) - pad] if pad else l)
    eps = float(np.percentile(losses_f, 95))
    print(f"noise floor eps (95%) = {eps:.4f}")
    print("  (注: fin 摂動の位相蓄積で eps は過大になりやすい。縮退判定は ratio を主とする)")

    # ---------------- 無変調ベースライン ----------------
    # b[t] = loss(target_t, 無変調音) = その変調の「効きの大きさ」。
    # ratio[t,c] = minloss[t,c]/b[t] が主診断:
    #   ratio ≪ 1 … 候補が真似できる（縮退の疑い）
    #   ratio ≈ 1 … 候補では真似できない（ただし b[t] 自体が小さければ「効きが弱い先」）
    x_base = synth_np(np.array([0.0], np.float32), dests[0])  # depth 0 = 無変調

    # ---------------- margin 地図 ----------------
    nD = len(dests)
    minloss = np.full((nD, nD), np.nan)
    argmin_d = np.full((nD, nD), np.nan)
    b_eff = np.full((nD,), np.nan)
    xtype_ratio = np.full((nD,), np.nan)
    curves = {}
    for ti, t_dest in enumerate(dests):
        if vca_on and t_dest == 0:
            continue
        x_t = synth_np(np.array([args.dtrue], np.float32), t_dest)
        b_eff[ti] = float(loss_vs(tf.constant(x_base), tf.constant(x_t)).numpy()[0])
        for ci, c_dest in enumerate(dests):
            if vca_on and c_dest == 0:
                continue
            ml, am, ls = sweep_minloss(x_t, c_dest)
            minloss[ti, ci] = ml
            argmin_d[ti, ci] = am
            if {t_dest, c_dest} <= {5, 15}:            # drive ペアの曲線は保存
                curves[(t_dest, c_dest)] = ls
        # cross-type margin（v7）: 同一変調先・逆タイプで真似できるか
        other = "lfo" if args.probe == "env" else "env"
        xm, _, _ = sweep_minloss(x_t, t_dest, probe=other)
        xtype_ratio[ti] = xm / max(b_eff[ti], 1e-9)
        print(f"  target {DEST_NAMES[t_dest]:9s} done  "
              f"diag minloss={minloss[ti, ti]:.4f} @d={argmin_d[ti, ti]:+.2f}  "
              f"effect b={b_eff[ti]:.3f}")

    margin = minloss - np.diag(minloss)[:, None]
    ratio = minloss / b_eff[:, None]
    eps_diag = float(np.nanmax(np.diag(minloss)))
    print(f"eps(diag; v6 rule) = {eps_diag:.4f}  (対角minlossの最大値=自己一致誤差の床)")

    # ---------------- 出力 ----------------
    names = [DEST_NAMES[d] for d in dests]
    for tag, mat in (("minloss", minloss), ("margin", margin), ("ratio", ratio)):
        with open(f"{args.out}_{tag}.csv", "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["target\\cand"] + names)
            for ti, nm in enumerate(names):
                w.writerow([nm] + [f"{v:.5f}" for v in mat[ti]])

    off = [(ratio[ti, ci], margin[ti, ci], b_eff[ti], names[ti], names[ci])
           for ti in range(nD) for ci in range(nD)
           if ti != ci and np.isfinite(ratio[ti, ci])]
    off.sort()
    print("\n縮退の疑いが強い順（ratio = minloss/効き。簡易掃引なので『小さい=確実に疑い』）:")
    for rv, mv, bv, tn, cn in off[:10]:
        flag = "  <-- DEGENERATE (ratio<0.1)" if rv < 0.1 else (
               "  <-- partial (ratio<0.3)" if rv < 0.3 else "")
        weak = "  [b小=効き弱い先]" if bv < 0.15 * np.nanmax(b_eff) else ""
        print(f"  target={tn:9s} cand={cn:9s} ratio={rv:6.3f} margin={mv:7.4f} b={bv:6.3f}{flag}{weak}")
    other = "LFO" if args.probe == "env" else "ENV"
    print(f"\ncross-type margin（同一変調先・{other} 候補で真似。小=型縮退の疑い）:")
    for ti, nm in enumerate(names):
        if np.isfinite(xtype_ratio[ti]):
            fl = "  <-- TYPE-DEGENERATE" if xtype_ratio[ti] < 0.1 else (
                 "  <-- partial" if xtype_ratio[ti] < 0.3 else "")
            print(f"  {nm:9s}: ratio={xtype_ratio[ti]:6.3f}{fl}")
    diag_ok = all(abs(argmin_d[i, i] - args.dtrue) < 1e-6
                  for i in range(nD) if np.isfinite(argmin_d[i, i]))
    print(f"\n対角チェック（正しい変調先の argmin == d_true）: {'OK' if diag_ok else 'NG'}")

    # ヒートマップ
    plt.figure(figsize=(1.0 + 0.6 * nD, 0.8 + 0.55 * nD))
    mm = np.ma.masked_invalid(ratio)
    plt.imshow(mm, cmap="viridis_r", vmin=0.0, vmax=1.2)
    plt.colorbar(label="ratio = minloss / effect size (dark = degenerate)")
    plt.xticks(range(nD), names, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(nD), names, fontsize=8)
    plt.xlabel("candidate (swept) destination")
    plt.ylabel("target (true) destination")
    plt.title(f"Degeneracy ratio map  (d_true={args.dtrue}, eps={eps:.3f}, sr={SR})")
    for ti in range(nD):
        for ci in range(nD):
            if ti != ci and np.isfinite(ratio[ti, ci]) and ratio[ti, ci] < 0.1:
                plt.text(ci, ti, "!", ha="center", va="center", color="red", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{args.out}_heatmap.png", dpi=130)
    plt.close()

    # drive ペア曲線
    if {5, 15} <= set(dests):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, tdest in zip(axes, (5, 15)):
            other = 15 if tdest == 5 else 5
            ax.plot(sweep, curves[(tdest, tdest)], lw=2, label=f"sweep {DEST_NAMES[tdest]} (correct)")
            ax.plot(sweep, curves[(tdest, other)], lw=2, ls="--",
                    label=f"sweep {DEST_NAMES[other]} (wrong)")
            ax.axvline(args.dtrue, color="k", ls=":", lw=1)
            ti_local = dests.index(tdest)
            ax.axhline(b_eff[ti_local], color="g", ls="--", lw=1, label="no-mod baseline b")
            ax.axhline(eps, color="r", ls=":", lw=1, label="eps (noise floor)")
            ax.set_xlabel("depth")
            ax.set_ylabel("MRS-STFT distance")
            ax.set_title(f"target: env -> {DEST_NAMES[tdest]}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{args.out}_drivepair.png", dpi=130)
        plt.close()

    print(f"\nsaved: {args.out}_margin.csv / _minloss.csv / _heatmap.png"
          + (" / _drivepair.png" if {5, 15} <= set(dests) else ""))
    print(f"total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
