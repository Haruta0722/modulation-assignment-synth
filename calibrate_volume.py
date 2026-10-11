r"""calibrate_volume.py — 音量の ENV の attack / release を秒でずらしたときの、ペアごとの距離を測る

  python calibrate_volume.py --n 800                       # 対数メル・MR-STFT
  python calibrate_volume.py --n 400 --jtfs                # ＋ JTFS（PNP の知覚距離）

聞き分けの限界の基準: 立ち上がり・終わりの減衰の時間の弁別閾 ≈ 25%（van Heuven & van den Broecke 1979、
McAdams 2019 に引用）。元の時間が 80 ms 以上で当てはまりやすい。原典の "decay" は鳴り終わりの減衰なので、
このシンセでは release にあたる。

対象（test）: 音量の列（level1 / level2）にだけつながっている ENV。
  attack : 元の attack ≥ 80 ms、ずらしても attack + hold + decay ≤ ノートオフ時刻
  release: 元の release ≥ 80 ms、sustain > 0.05、ずらしても release ≤ 音の残り（1.5 − 1.0 秒）
ずらし方: attack または release だけを秒で ×f（f = 1/1.5, 1/1.25, 1/1.1, 1.1, 1.25, 1.5）。他の秒数はそのまま。
深さの条件:
  d1   : その ENV がつながる音量の列の深さを 1、静的な値を 0 にする（無音から最大まで、原典に近い条件）
  orig : 元の深さのまま（|深さ| の区間別に集計）
距離（元の音と、ずらした音のペアごと）
  logmel   : 対数メル（モデルの入力と同じ）の平均絶対差
  mrstft   : MR-STFT 損失
  jtfs_pnp : JTFS（PNP 損失の M(θ) を作ったのと同じ設定 J=8, Q=6, J_fr=2, Q_fr=2）の係数の差の L2
  jtfs_slow: JTFS の時間方向の 2 次ウェーブレットを約 1 Hz まで伸ばした設定（J_fr=6）の L2
  *_rel    : JTFS の差の L2 を元の音の係数の L2 で割ったもの（音の大きさによらない）
知覚的に一様な距離なら、attack ±25% と release ±25% の距離はほぼ等しくなるはず。
"""
import argparse, os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import train as T
from audibility import logmel, quantize
from diff_synth_fast import synth_fast
from jtfs_tf import JTFS

K = ds.K_ENVS
LEVEL = {0, 1}                                       # level1, level2
FACTORS = [1 / 1.5, 1 / 1.25, 1 / 1.1, 1.1, 1.25, 1.5]
DEPTH_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v7")
    ap.add_argument("--n", type=int, default=800, help="対象の音の最大数")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--jtfs", action="store_true", help="JTFS の距離も測る（重い）")
    ap.add_argument("--modes", default="d1,orig")
    ap.add_argument("--out", default="calibrate_volume.npz")
    args = ap.parse_args()

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    TAIL = N / SR - T_OFF

    # 距離: 名前 → (特徴を作る関数, 2 つの特徴から距離 [B] を出す関数)
    feats = {"logmel": (lambda x: logmel(x, SR),
                        lambda a, b: tf.reduce_mean(tf.abs(a - b), [1, 2]).numpy())}
    if args.jtfs:
        for name, kw in (("jtfs_pnp", dict(J=8, Q=6, J_fr=2, Q_fr=2)),
                         ("jtfs_slow", dict(J=8, Q=6, J_fr=6, Q_fr=2))):
            phi = JTFS(sr=SR, **kw)
            f = tf.function(phi.call, reduce_retracing=True)
            feats[name] = (f, lambda a, b: tf.norm(a - b, axis=1).numpy())
            feats[name + "_rel"] = (None, None)      # name の特徴を使って割合を出す

    # 対象の音と、その音量の ENV（ソース番号）を選ぶ
    P = data.P
    picks = []                                       # (データの通し番号, ソース k, attack 対象か, release 対象か)
    for i in data.splits["test"]:
        z, sp, st = P["z"][i].astype(int), P["src_params"][i], P["src_type"][i]
        for k in range(K):
            cols = set(np.nonzero(z == k)[0])
            if not cols or st[k] != 1 or not cols <= LEVEL:
                continue
            a, h, d, s, r = sp[k]
            okA, okR = a >= 0.08, r >= 0.08 and s > 0.05
            if okA or okR:
                picks.append((i, k, okA, okR))
                break
        if len(picks) >= args.n:
            break
    print(f"対象の音 {len(picks)}（attack {sum(p[2] for p in picks)} / release {sum(p[3] for p in picks)}）"
          f"  距離: {', '.join(feats)}", flush=True)

    def synth(b, sp, R, th):
        f0 = b["f0"].numpy()
        out = np.zeros((len(f0), N), np.float32)
        for f in np.unique(f0):
            s = np.nonzero(f0 == f)[0]
            fmax = float(f) * 2 ** (100 / 1200)
            hs, hq = int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2)
            g = lambda t: tf.gather(tf.constant(t) if isinstance(t, np.ndarray) else t, s)
            out[s] = synth_fast(g(th), g(b["theta_osc"]), g(b["gmk"]), g(sp), g(R), g(b["f0"]),
                                T_OFF, N, SR, hs, hq, src_type=g(b["src_type"])).numpy()
        return quantize(tf.constant(out), data.scale)

    def features(x):
        return {k: fn(x) for k, (fn, _) in feats.items() if fn is not None}

    def distances(F0, x0, x1):
        F1 = features(x1)
        out = {k: feats[k][1](F0[k], F1[k]) for k in F0}
        out["mrstft"] = ds.mrstft_loss(x1, x0).numpy()
        for k in list(F0):
            if k.startswith("jtfs"):
                out[k + "_rel"] = out[k] / np.maximum(tf.norm(F0[k], axis=1).numpy(), 1e-12)
        return out

    rows = []                                        # dict: param, f, mode, depth, 各距離
    modes = args.modes.split(",")
    for s0 in range(0, len(picks), args.batch):
        chunk = picks[s0:s0 + args.batch]
        idx = np.array([p[0] for p in chunk])
        ks = np.array([p[1] for p in chunk])
        x, b = data.batch(idx)
        sp0, R0, th0 = b["src_params"].numpy(), b["R"].numpy(), b["theta_stat"].numpy()
        z = b["z"].numpy()
        B = len(idx)
        dep = np.array([np.abs(R0[j, ks[j], z[j] == ks[j]]).max() for j in range(B)])

        # 深さ 1 の条件: 音量の列の深さを 1、静的な値を 0 に
        R1, th1 = R0.copy(), th0.copy()
        for j in range(B):
            for c in np.nonzero(z[j] == ks[j])[0]:
                R1[j, ks[j], c], th1[j, c] = 1.0, 0.0

        for mode, R, th in (("d1", R1, th1), ("orig", R0, th0)):
            if mode not in modes:
                continue
            x0 = synth(b, sp0, R, th)
            F0 = features(x0)
            for param in ("attack", "release"):
                for f in FACTORS:
                    sp = sp0.copy()
                    ok = np.zeros(B, bool)
                    for j, (_, k, okA, okR) in enumerate(chunk):
                        a, h, d, s, r = sp0[j, k]
                        if param == "attack" and okA and a * f + h + d <= T_OFF:
                            sp[j, k, 0], ok[j] = a * f, True
                        if param == "release" and okR and r * f <= TAIL:
                            sp[j, k, 4], ok[j] = r * f, True
                    if not ok.any():
                        continue
                    D = distances(F0, x0, synth(b, sp, R, th))
                    for j in np.nonzero(ok)[0]:
                        rows.append(dict(param=param, f=f, mode=mode, depth=dep[j],
                                         **{k: float(v[j]) for k, v in D.items()}))
        print(f"  {s0 + B} / {len(picks)}", flush=True)

    names = [k for k in rows[0] if k not in ("param", "f", "mode", "depth")]
    col = lambda k: np.array([r[k] for r in rows])
    par, mod, fac, dep = col("param"), col("mode"), col("f").astype(float), col("depth").astype(float)
    jnd = np.isclose(np.abs(np.log(fac)), np.log(1.25))

    print("\n==== 弁別閾（±25%）での距離の中央値と、attack / release の比（1 に近いほど知覚的に一様） ====")
    for mode in modes:
        print(f"\n  [{mode}]")
        print(f"  {'距離':14s}{'attack':>10s}{'release':>10s}{'比 A/R':>9s}   距離と |log 倍率| の順位相関（単調性）")
        for k in names:
            v = col(k).astype(float)
            sa = (par == "attack") & (mod == mode) & jnd
            sr = (par == "release") & (mod == mode) & jnd
            ma, mr = np.median(v[sa]), np.median(v[sr])
            sm = mod == mode
            rk = lambda u: np.argsort(np.argsort(u))
            rho = np.corrcoef(rk(v[sm]), rk(np.abs(np.log(fac[sm]))))[0, 1]
            print(f"  {k:14s}{ma:10.4g}{mr:10.4g}{ma / mr:9.2f}   {rho:.3f}")

    print("\n==== 倍率ごとの中央値 ====")
    for mode in modes:
        for param in ("attack", "release"):
            print(f"\n  [{mode} / {param}]  " + "  ".join(f"×{f:.3f}" for f in FACTORS))
            for k in names:
                v = col(k).astype(float)
                vals = []
                for f in FACTORS:
                    s = (par == param) & (mod == mode) & np.isclose(fac, f)
                    vals.append(f"{np.median(v[s]):.4g}" if s.any() else "-")
                print(f"  {k:14s}" + "  ".join(f"{t:>7s}" for t in vals))

    if "orig" in modes:
        print("\n==== 元の深さ・±25% の距離の中央値を |深さ| の区間別に ====")
        for k in names:
            v = col(k).astype(float)
            for param in ("attack", "release"):
                s25 = (par == param) & (mod == "orig") & jnd
                line = [f"{np.median(v[s25 & (dep >= lo) & (dep < hi)]):.4g}"
                        if (s25 & (dep >= lo) & (dep < hi)).any() else "-" for lo, hi in DEPTH_BINS]
                print(f"  {k:14s}{param:8s} " + " / ".join(line))

    np.savez(args.out, param=par, mode=mod, factor=fac, depth=dep,
             **{k: col(k).astype(float) for k in names})


if __name__ == "__main__":
    main()
