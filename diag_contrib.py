"""
diag_contrib.py — 各パラメータの推定誤差が音に与える影響を測る
================================================================================
背景:
  静的パラメータの診断（diag_static.py）で、20 個中 5 個が実質推定できていない
  （res_q, gmk, osc_a1/a2, ratio など MAE ≈ 0.23）。
  これが「音を壊すほど致命的か」を確かめたい。

  対角知覚感度（precompute_Mdiag.py）は**微小変化に対する微分**を測るもので、
  MAE 0.23 という有限の誤差の影響を必ずしも予測しない
  （局所線形近似が有効な範囲でしか対応しない）。
  実際、level は感度上位なのに shape_ok は最下位クラスという乖離が観測されている。
  → 感度（微分）と実誤差の影響（有限差分）を**両方測り、一致するかを見る**。

方法:
  1. 真値パラメータのパッチから音 x_true を合成
  2. パラメータを **1 つだけ**、推定誤差に相当する量だけずらして音 x_pert を合成
     ずらす量は diag_static.py の実測 MAE（--use_measured）または指定値
  3. 音響距離 MRS-STFT(x_true, x_pert) を測る = そのパラメータの誤差が音に与える影響
  4. 対角知覚感度との相関を見る

  比較のため「全パラメータを同時にずらした場合」も測り、
  個別寄与の和との差から相互作用の大きさも分かる。

使い方:
  python diag_contrib.py --n 64 --use_measured
  python diag_contrib.py --n 64 --delta 0.23      # 一律 0.23 ずらす
"""
import argparse
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast
from diag_static import make_batch

K, M = ds.K_ENVS, ds.M_DEST

# diag_static.py の実測 MAE（mod=full, 3000 steps）
MEASURED = {
    "cutoff": 0.0605, "fin1": 0.0741, "fin2": 0.0743, "thresh": 0.1310,
    "eq_high": 0.1315, "eq_mid": 0.1520, "rate": 0.1548, "ratio": 0.1953,
    "level1": 0.1977, "eq_low": 0.2018, "drive_f": 0.2045, "depth": 0.2047,
    "feedback": 0.2066, "res_q": 0.2082, "level2": 0.2087, "mix_d": 0.2130,
    "drive_d": 0.2132, "osc_a1": 0.2323, "osc_a2": 0.2338, "gmk": 0.2357,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--delta", type=float, default=0.0,
                    help="一律のずらし量。0 なら --use_measured を使う")
    ap.add_argument("--use_measured", action="store_true",
                    help="diag_static の実測 MAE をずらし量に使う")
    ap.add_argument("--mod", choices=["none", "single", "full"], default="full")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--toff", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    SR, N, T_OFF = args.sr, int(args.sr * args.T), args.toff
    fmax = 65.4 * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    rng = np.random.default_rng(args.seed)
    names = [s[0] for s in ds.dest_specs(SR)] + ["osc_a1", "osc_a2", "gmk"]
    P = M + 3

    @tf.function(reduce_retracing=True)
    def synth(th, osc, gmk, src, R, f0, styp):
        return synth_fast(th, osc, gmk, src, R, f0, T_OFF, N, SR,
                          NH[0], NH[1], src_type=styp)

    def deltas_for(i):
        if args.delta > 0:
            return args.delta
        return MEASURED.get(names[i], 0.2)

    contrib = np.zeros((0, P))
    joint = []
    for s0 in range(0, args.n, args.batch):
        b = min(args.batch, args.n - s0)
        th, osc, gmk, src, styp, R, f0 = make_batch(rng, b, args.mod, SR, N, T_OFF)
        args_t = [tf.constant(v) for v in (th, osc, gmk, src, R, f0, styp)]
        x0 = synth(*args_t)

        rows = np.zeros((b, P), np.float32)
        for i in range(P):
            d = deltas_for(i)
            th2, osc2, gmk2 = th.copy(), osc.copy(), gmk.copy()
            sign = np.where(rng.random(b) < 0.5, 1.0, -1.0).astype(np.float32)
            if i < M:
                th2[:, i] = np.clip(th2[:, i] + sign * d, 0.0, 1.0)
            elif i < M + 2:
                osc2[:, i - M] = np.clip(osc2[:, i - M] + sign * d, 0.0, 1.0)
            else:
                gmk2 = np.clip(gmk2 + sign * d, 0.0, 1.0)
            xp = synth(tf.constant(th2), tf.constant(osc2), tf.constant(gmk2),
                       args_t[3], args_t[4], args_t[5], args_t[6])
            rows[:, i] = ds.mrstft_loss(x0, xp).numpy()
        contrib = np.concatenate([contrib, rows], 0)

        # 全パラメータ同時（相互作用の確認）
        sg = np.where(rng.random((b, P)) < 0.5, 1.0, -1.0).astype(np.float32)
        dv = np.array([deltas_for(i) for i in range(P)], np.float32)
        thj = np.clip(th + sg[:, :M] * dv[:M], 0, 1)
        oscj = np.clip(osc + sg[:, M:M+2] * dv[M:M+2], 0, 1)
        gmkj = np.clip(gmk + sg[:, M+2] * dv[M+2], 0, 1)
        xj = synth(tf.constant(thj), tf.constant(oscj), tf.constant(gmkj),
                   args_t[3], args_t[4], args_t[5], args_t[6])
        joint += list(ds.mrstft_loss(x0, xj).numpy())
        print(f"  {min(s0+b, args.n)}/{args.n}", flush=True)

    c = contrib.mean(0)
    order = np.argsort(-c)
    print(f"\n== 推定誤差が音に与える影響（MRS-STFT 距離、mod={args.mod}） ==")
    print(f"{'パラメータ':12s} {'音響影響':>9s} {'ずらし量':>9s} {'影響/誤差':>10s}")
    for i in order:
        d = deltas_for(i)
        print(f"  {names[i]:10s} {c[i]:9.4f} {d:9.4f} {c[i]/max(d,1e-6):10.3f}")

    print(f"\n  個別寄与の合計 = {c.sum():.4f}")
    print(f"  全同時ずらし    = {np.mean(joint):.4f}")
    print(f"  → 合計より小さければ、誤差どうしが打ち消し合っている（相互作用）")

    # 「推定できていない」群と「できている」群の比較
    bad = [i for i in range(P) if deltas_for(i) >= 0.20]
    good = [i for i in range(P) if deltas_for(i) < 0.15]
    print(f"\n  推定できている群（MAE<0.15, n={len(good)}）の音響影響 平均 "
          f"= {c[good].mean():.4f}")
    print(f"  推定できていない群（MAE≥0.20, n={len(bad)}）の音響影響 平均 "
          f"= {c[bad].mean():.4f}")
    print(f"  → 後者が小さければ「読めないパラメータは音への影響も小さい」")

    np.savez("contrib.npz", contrib=contrib, names=np.array(names),
             joint=np.array(joint))
    print("\nsaved: contrib.npz")


if __name__ == "__main__":
    main()
