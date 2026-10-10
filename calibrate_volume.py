r"""calibrate_volume.py — 音量の ENV の attack / release を秒でずらしたときの、ペアごとの距離を測る

  python calibrate_volume.py --n 800

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
距離（元の音と、ずらした音のペアごと）: 対数メル（モデルの入力と同じ）の平均絶対差、MR-STFT 損失。
"""
import argparse, os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import train as T
from audibility import logmel, quantize
from diff_synth_fast import synth_fast

K = ds.K_ENVS
LEVEL = {0, 1}                                       # level1, level2
FACTORS = [1 / 1.5, 1 / 1.25, 1 / 1.1, 1.1, 1.25, 1.5]
DEPTH_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v7")
    ap.add_argument("--n", type=int, default=800, help="対象の音の最大数")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    TAIL = N / SR - T_OFF

    # 対象の音と、その音量の ENV（ソース番号）を選ぶ
    idx_all = data.splits["test"]
    P = data.P
    picks = []                                       # (データの通し番号, ソース k, attack 対象か, release 対象か)
    for i in idx_all:
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
    print(f"対象の音 {len(picks)}（attack {sum(p[2] for p in picks)} / release {sum(p[3] for p in picks)}）", flush=True)

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

    rows = []                                        # (param, f, mode, |深さ|, 対数メル, MR-STFT)
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
            x0 = synth(b, sp0, R, th)
            L0 = logmel(x0, SR)
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
                    x1 = synth(b, sp, R, th)
                    dm = tf.reduce_mean(tf.abs(logmel(x1, SR) - L0), [1, 2]).numpy()
                    dr = ds.mrstft_loss(x1, x0).numpy()
                    for j in np.nonzero(ok)[0]:
                        rows.append((param, f, mode, dep[j], dm[j], dr[j]))
        print(f"  {s0 + B} / {len(picks)}", flush=True)

    rec = np.array([(r[1], r[3], r[4], r[5]) for r in rows])
    par = np.array([r[0] for r in rows])
    mod = np.array([r[2] for r in rows])
    q = lambda v: " / ".join(f"{x:.4f}" for x in np.percentile(v, [25, 50, 75]))

    for mode, label in (("d1", "深さ 1（静的な値 0）"), ("orig", "元の深さ")):
        print(f"\n==== {label} ====")
        for param in ("attack", "release"):
            print(f"\n  {param}")
            print(f"  {'倍率':>8s}{'件数':>6s}   {'対数メル(25/50/75%)':26s}   MR-STFT(25/50/75%)")
            for f in FACTORS:
                s = (par == param) & (mod == mode) & np.isclose(rec[:, 0], f)
                if s.sum():
                    tag = " ← 弁別閾" if np.isclose(abs(np.log(f)), np.log(1.25)) else ""
                    print(f"  ×{f:6.3f}{s.sum():6d}   {q(rec[s, 2]):26s}   {q(rec[s, 3])}{tag}")

    print("\n==== 元の深さ・±25% の距離（対数メルの中央値）を |深さ| の区間別に ====")
    for param in ("attack", "release"):
        s25 = (par == param) & (mod == "orig") & np.isclose(np.abs(np.log(rec[:, 0])), np.log(1.25))
        line = []
        for lo, hi in DEPTH_BINS:
            s = s25 & (rec[:, 1] >= lo) & (rec[:, 1] < hi)
            line.append(f"[{lo:.1f},{hi:.1f}) n={s.sum()} {np.median(rec[s, 2]) if s.any() else float('nan'):.4f}")
        print(f"  {param:8s} " + "   ".join(line))

    np.savez("calibrate_volume.npz", factor=rec[:, 0], depth=rec[:, 1], logmel=rec[:, 2],
             mrstft=rec[:, 3], param=par, mode=mod)


if __name__ == "__main__":
    main()
