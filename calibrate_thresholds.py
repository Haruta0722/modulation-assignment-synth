r"""calibrate_thresholds.py — 評価のしきい値を「聞き分けられる差」に合わせて較正する

  python calibrate_thresholds.py --n 384

真のパラメータで音を作り直し（生成時と同じ合成・int16 量子化）、1 か所だけ誤差を入れた音との距離を測る。
  基準     : ENV のアタック時間を 25% 長く/短く（立ち上がり時間の弁別閾 ≈ 25%、McAdams 2019）
  深さ     : そのソースがつながる列の 1 つの深さを δ ずらす
  ENV 形状 : ENV の 5 つの割合（attack/T, hold/残り, decay/残り, sustain, release/R_MAX）に正規雑音
  LFO 位相 : 位相を Δφ（周期に対する割合）ずらす
  LFO レート: レートを × r
距離はモデルの入力と同じ対数メルの平均絶対差（主）と MR-STFT。各誤差の大きさごとに、距離が基準
（アタック 25% の距離の中央値）を超える割合を出し、中央値が基準に達する誤差をしきい値の目安とする。
形状の誤差は評価と同じく、ソース信号 e(n) のフレーム上の平均絶対差。
"""
import argparse, os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import estimator as E
import train as T
from audibility import logmel, quantize
from diff_synth_fast import synth_fast

K = ds.K_ENVS
DEPTHS = [0.05, 0.1, 0.15, 0.2, 0.3]
SIGMAS = [0.03, 0.06, 0.1, 0.15, 0.2, 0.3]
PHASES = [0.05, 0.1, 0.25, 0.5]
RATES = [1.05, 1.1, 1.25, 1.5]
SHAPE_BINS = [0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 1.0]


def to_frac(p, t_off):
    a, h, d, s, r = p
    return np.array([a / t_off, h / max(t_off - a, 1e-3), d / max(t_off - a - h, 1e-3), s, r / E.R_MAX])


def from_frac(u, t_off):
    u = np.clip(u, 0.0, 1.0)
    a = max(u[0] * t_off, 1e-3)
    h = u[1] * (t_off - a)
    d = max(u[2] * (t_off - a - h), 1e-3)
    return np.array([a, h, d, u[3], max(u[4] * E.R_MAX, 1e-3)], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v7")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    n_frames = 1 + (N - 1024) // 256
    fidx = np.minimum(np.arange(n_frames) * 256 + 512, N - 1)

    def synth(b, sp, R):
        f0 = b["f0"].numpy()
        out = np.zeros((len(f0), N), np.float32)
        for f in np.unique(f0):
            s = np.nonzero(f0 == f)[0]
            fmax = float(f) * 2 ** (100 / 1200)
            hs, hq = int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2)
            g = lambda t: tf.gather(t, s)
            out[s] = synth_fast(g(b["theta_stat"]), g(b["theta_osc"]), g(b["gmk"]),
                                g(tf.constant(sp)), g(tf.constant(R)), g(b["f0"]), T_OFF, N, SR,
                                hs, hq, src_type=g(b["src_type"])).numpy()
        return quantize(tf.constant(out), data.scale)

    def src_sig(sp, styp):
        return ds.sources(tf.constant(sp), tf.constant(styp), T_OFF, N, SR).numpy()[:, :, fidx]

    rec = {}                                                          # name → list of (誤差, 対数メル, MR-STFT)

    def add(name, err, dmel, dmr, mask):
        for e, a, b_ in zip(np.atleast_1d(err)[mask] if np.ndim(err) else [err] * mask.sum(),
                            dmel[mask], dmr[mask]):
            rec.setdefault(name, []).append((float(e), float(a), float(b_)))

    idx_all = data.splits["test"][:args.n]
    for s0 in range(0, len(idx_all), args.batch):
        idx = idx_all[s0:s0 + args.batch]
        x, b = data.batch(idx)
        sp0, R0 = b["src_params"].numpy(), b["R"].numpy()
        styp, z = b["src_type"].numpy(), b["z"].numpy()
        B = len(idx)
        used = np.stack([(z == k).any(1) for k in range(K)], 1)
        k_env = np.array([np.argmax(used[i] & (styp[i] == 1)) if (used[i] & (styp[i] == 1)).any() else -1
                          for i in range(B)])
        k_lfo = np.array([np.argmax(used[i] & (styp[i] == 0)) if (used[i] & (styp[i] == 0)).any() else -1
                          for i in range(B)])
        has_env, has_lfo = k_env >= 0, k_lfo >= 0
        x0 = synth(b, sp0, R0)
        L0 = logmel(x0, SR)
        e0 = src_sig(sp0, styp)

        def dist(sp=None, R=None):
            x1 = synth(b, sp0 if sp is None else sp, R0 if R is None else R)
            return (tf.reduce_mean(tf.abs(logmel(x1, SR) - L0), [1, 2]).numpy(),
                    ds.mrstft_loss(x1, x0).numpy())

        def shape_err(sp, k):
            e1 = src_sig(sp, styp)
            return np.array([np.abs(e1[i, k[i]] - e0[i, k[i]]).mean() if k[i] >= 0 else np.nan
                             for i in range(B)])

        # 基準: アタック ±25%（長くする / 短くするを半々）
        sp = sp0.copy()
        for i in np.nonzero(has_env)[0]:
            u = to_frac(sp0[i, k_env[i]], T_OFF)
            fac = 1.25 if rng.random() < 0.5 else 1 / 1.25
            u[0] = min(u[0] * fac, 0.98)
            sp[i, k_env[i]] = from_frac(u, T_OFF)
        dm, dr = dist(sp=sp)
        add("基準 attack±25%", shape_err(sp, k_env), dm, dr, has_env)

        # 深さ
        for dlt in DEPTHS:
            R = R0.copy()
            for i in np.nonzero(has_env)[0]:
                cols = np.nonzero(z[i] == k_env[i])[0]
                c = cols[np.argmax(np.abs(R0[i, k_env[i], cols]))]
                v = R0[i, k_env[i], c]
                R[i, k_env[i], c] = v - np.sign(v) * dlt if abs(v) >= dlt else v + np.sign(v) * dlt
            dm, dr = dist(R=R)
            add(f"深さ δ={dlt}", np.full(B, dlt), dm, dr, has_env)

        # ENV の形状（割合に正規雑音）
        for sig in SIGMAS:
            sp = sp0.copy()
            for i in np.nonzero(has_env)[0]:
                u = to_frac(sp0[i, k_env[i]], T_OFF) + rng.normal(0, sig, 5)
                sp[i, k_env[i]] = from_frac(u, T_OFF)
            dm, dr = dist(sp=sp)
            add("ENV 形状", shape_err(sp, k_env), dm, dr, has_env)

        # LFO の位相・レート
        for dp in PHASES:
            sp = sp0.copy()
            for i in np.nonzero(has_lfo)[0]:
                sp[i, k_lfo[i], 1] = (sp0[i, k_lfo[i], 1] + dp) % 1.0
            dm, dr = dist(sp=sp)
            add(f"LFO 位相 Δφ={dp}", shape_err(sp, k_lfo), dm, dr, has_lfo)
        for rr in RATES:
            sp = sp0.copy()
            for i in np.nonzero(has_lfo)[0]:
                sp[i, k_lfo[i], 0] = np.clip(sp0[i, k_lfo[i], 0] * rr, 0.1, 10.0)
            dm, dr = dist(sp=sp)
            add(f"LFO レート ×{rr}", shape_err(sp, k_lfo), dm, dr, has_lfo)
        print(f"  {s0 + B} / {len(idx_all)}", flush=True)

    base = np.array(rec["基準 attack±25%"])
    jnd_mel, jnd_mr = np.median(base[:, 1]), np.median(base[:, 2])
    print(f"\n基準（アタック ±25%）: 件数 {len(base)}  対数メル距離の中央値 {jnd_mel:.4f}  "
          f"MR-STFT の中央値 {jnd_mr:.4f}  （このときの形状の誤差の中央値 {np.median(base[:, 0]):.3f}）")

    print(f"\n{'誤差の種類':22s}{'件数':>6s}{'形状の誤差(中央値)':>18s}{'対数メル(中央値)':>16s}{'基準を超える割合':>16s}")
    for name in [k for k in rec if k != "ENV 形状" and k != "基準 attack±25%"]:
        a = np.array(rec[name])
        print(f"{name:22s}{len(a):6d}{np.nanmedian(a[:, 0]):18.3f}{np.median(a[:, 1]):16.4f}"
              f"{100 * np.mean(a[:, 1] > jnd_mel):15.1f}%")

    a = np.array(rec["ENV 形状"])
    a = a[~np.isnan(a[:, 0])]
    print(f"\nENV の形状の誤差ごと（件数 {len(a)}）")
    print(f"{'形状の誤差':14s}{'件数':>6s}{'対数メル(中央値)':>16s}{'基準を超える割合':>16s}")
    for lo, hi in zip(SHAPE_BINS[:-1], SHAPE_BINS[1:]):
        s = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if s.sum() >= 10:
            print(f"{lo:.3f}〜{hi:.3f}  {s.sum():6d}{np.median(a[s, 1]):16.4f}{100 * np.mean(a[s, 1] > jnd_mel):15.1f}%")
    np.savez("calibrate_thresholds.npz", **{k: np.array(v) for k, v in rec.items()},
             jnd_mel=jnd_mel, jnd_mr=jnd_mr)


if __name__ == "__main__":
    main()
