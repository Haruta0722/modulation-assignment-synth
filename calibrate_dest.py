r"""calibrate_dest.py — 変調先ごとに、見かけの変調 d·e(n) の許容誤差 τ_m を較正する

  python calibrate_dest.py --ref calibrate_volume_s08.npz --per_dest 80

desc_rec で問われるのは、変調先 m の値が時間とともにどう動くか（見かけの変調 d·e(n)）。
変調先 m だけが変わる誤差を入れた音と元の音のペアで知覚距離を測り、d·e(n) の誤差（フレーム上の平均絶対差）
との関係から、距離が聞き分けの限界 D*（音量の attack / release ±25% の距離、calibrate_volume.py）に届く
d·e の誤差を τ_m とする。

誤差の入れ方（変調先 m の列だけが変わるもの）
  depth: 列 m の深さを ±δ（|深さ| が小さくなる向き。δ > |深さ| なら大きくなる向き）
  env  : 列 m の ENV が m にしかつながっていない音で、割合（attack/T, hold/残り, decay/残り, sustain, release/R_MAX）に正規雑音
  lfo  : 列 m の LFO が m にしかつながっていない音で、位相を Δφ ずらす / レートを ×r
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
from jtfs_tf import JTFS

K, M = ds.K_ENVS, ds.M_DEST
NAMES = [d[0] for d in ds.dest_specs(16000)]
DEPTHS = [0.025, 0.05, 0.1, 0.2]
SIGMAS = [0.03, 0.06, 0.1, 0.2]
PHASES = [0.05, 0.1, 0.25]
RATES = [1.05, 1.1, 1.25]


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
    ap.add_argument("--ref", default="calibrate_volume_s08.npz", help="聞き分けの限界 D* を取る較正結果")
    ap.add_argument("--ref_mode", default="d1")
    ap.add_argument("--per_dest", type=int, default=80, help="変調先ごとの音の数")
    ap.add_argument("--splits", default="test,val")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="calibrate_dest.npz")
    ap.add_argument("--antialias", action="store_true", help="JTFS で間引く前に平均する（折り返しを防ぐ）")
    ap.add_argument("--dests", default="", help="変調先の番号を絞る（例: 2,3,4,7,10）")
    ap.add_argument("--depths", default="", help="深さの誤差（例: 0.003,0.006,0.012,0.025）")
    ap.add_argument("--sigmas", default="", help="ENV の割合の雑音（例: 0.005,0.01,0.02,0.04）")
    args = ap.parse_args()
    global DEPTHS, SIGMAS
    if args.depths:
        DEPTHS = [float(v) for v in args.depths.split(",")]
    if args.sigmas:
        SIGMAS = [float(v) for v in args.sigmas.split(",")]
    dests = [int(v) for v in args.dests.split(",")] if args.dests else list(range(M))
    rng = np.random.default_rng(args.seed)

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    fidx = np.minimum(np.arange(1 + (N - 1024) // 256) * 256 + 512, N - 1)

    # 距離（calibrate_volume.py と同じもの）
    phis = {"jtfs_pnp": JTFS(sr=SR, J=8, Q=6, J_fr=2, Q_fr=2, antialias=args.antialias),
            "jtfs_slow": JTFS(sr=SR, J=8, Q=6, J_fr=6, Q_fr=2, antialias=args.antialias)}
    fns = {k: tf.function(p.call, reduce_retracing=True) for k, p in phis.items()}

    def features(x):
        F = {"logmel": logmel(x, SR)}
        F.update({k: f(x) for k, f in fns.items()})
        return F

    def distances(F0, x1):
        F1 = features(x1)
        out = {"logmel": tf.reduce_mean(tf.abs(F1["logmel"] - F0["logmel"]), [1, 2]).numpy()}
        for k in fns:
            d = tf.norm(F1[k] - F0[k], axis=1).numpy()
            out[k] = d
            out[k + "_rel"] = d / np.maximum(tf.norm(F0[k], axis=1).numpy(), 1e-12)
        return out

    # 聞き分けの限界 D*: 音量の attack / release を ±25% 変えたときの距離の中央値
    ref = np.load(args.ref)
    jnd = np.isclose(np.abs(np.log(ref["factor"])), np.log(1.25)) & (ref["mode"] == args.ref_mode)
    dist_names = [k for k in ref.files if k not in ("param", "mode", "factor", "depth", "mrstft")]
    Dstar = {k: dict(attack=float(np.median(ref[k][jnd & (ref["param"] == "attack")])),
                     release=float(np.median(ref[k][jnd & (ref["param"] == "release")])),
                     both=float(np.median(ref[k][jnd]))) for k in dist_names}
    print("聞き分けの限界 D*（音量 ±25% の距離の中央値）")
    for k, v in Dstar.items():
        print(f"  {k:14s} attack {v['attack']:.4g}  release {v['release']:.4g}  両方 {v['both']:.4g}"
              f"  比 {v['attack'] / v['release']:.2f}")

    # 変調先ごとに対象の音を選ぶ（列 m に割り当てがある音。ENV / LFO の誤差はソースが m にしかつながらない音のみ）
    P = data.P
    pool = np.concatenate([data.splits[s] for s in args.splits.split(",")])
    pool = pool[rng.permutation(len(pool))]
    pairs = []                                       # (データの通し番号, m, k, 単独でつながるか)
    for m in dests:
        cnt = 0
        for i in pool:
            z = P["z"][i].astype(int)
            k = z[m]
            if k < 0:
                continue
            pairs.append((i, m, k, int((z == k).sum()) == 1))
            cnt += 1
            if cnt >= args.per_dest:
                break
    print(f"\n対象の (音, 変調先) {len(pairs)} 組", flush=True)

    def synth(b, sp, R):
        f0 = b["f0"].numpy()
        out = np.zeros((len(f0), N), np.float32)
        for f in np.unique(f0):
            s = np.nonzero(f0 == f)[0]
            fmax = float(f) * 2 ** (100 / 1200)
            hs, hq = int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2)
            g = lambda t: tf.gather(tf.constant(t) if isinstance(t, np.ndarray) else t, s)
            out[s] = synth_fast(g(b["theta_stat"]), g(b["theta_osc"]), g(b["gmk"]), g(sp), g(R), g(b["f0"]),
                                T_OFF, N, SR, hs, hq, src_type=g(b["src_type"])).numpy()
        return quantize(tf.constant(out), data.scale)

    def de(sp, R, styp, ms):
        """列 ms[j] の見かけの変調 d·e(n)（フレーム上）"""
        e = ds.sources(tf.constant(sp), tf.constant(styp), T_OFF, N, SR).numpy()[:, :, fidx]   # [B,K,T]
        mod = np.einsum("bkm,bkt->bmt", R, e)
        return mod[np.arange(len(ms)), ms]                                                  # [B,T]

    rows = []
    for s0 in range(0, len(pairs), args.batch):
        chunk = pairs[s0:s0 + args.batch]
        idx = np.array([p[0] for p in chunk])
        ms = np.array([p[1] for p in chunk])
        ks = np.array([p[2] for p in chunk])
        alone = np.array([p[3] for p in chunk])
        x, b = data.batch(idx)
        sp0, R0, styp = b["src_params"].numpy(), b["R"].numpy(), b["src_type"].numpy()
        B = len(idx)
        is_env = styp[np.arange(B), ks] == 1
        de0 = de(sp0, R0, styp, ms)
        x0 = synth(b, sp0, R0)
        F0 = features(x0)

        trials = []                                  # (種類, 大きさ, sp, R, 有効な行)
        for dl in DEPTHS:
            R = R0.copy()
            for j in range(B):
                v = R0[j, ks[j], ms[j]]
                R[j, ks[j], ms[j]] = v - np.sign(v) * dl if abs(v) >= dl else v + np.sign(v) * dl
            trials.append(("depth", dl, sp0, R, np.ones(B, bool)))
        for sg in SIGMAS:
            sp = sp0.copy()
            for j in np.nonzero(alone & is_env)[0]:
                sp[j, ks[j]] = from_frac(to_frac(sp0[j, ks[j]], T_OFF) + rng.normal(0, sg, 5), T_OFF)
            trials.append(("env", sg, sp, R0, alone & is_env))
        for dp in PHASES:
            sp = sp0.copy()
            for j in np.nonzero(alone & ~is_env)[0]:
                sp[j, ks[j], 1] = (sp0[j, ks[j], 1] + dp) % 1.0
            trials.append(("lfo_phase", dp, sp, R0, alone & ~is_env))
        for rr in RATES:
            sp = sp0.copy()
            for j in np.nonzero(alone & ~is_env)[0]:
                sp[j, ks[j], 0] = np.clip(sp0[j, ks[j], 0] * rr, 0.1, 10.0)
            trials.append(("lfo_rate", rr, sp, R0, alone & ~is_env))

        for kind, mag, sp, R, ok in trials:
            if not ok.any():
                continue
            err = np.abs(de(sp, R, styp, ms) - de0).mean(-1)
            D = distances(F0, synth(b, sp, R))
            for j in np.nonzero(ok)[0]:
                rows.append(dict(m=int(ms[j]), kind=kind, mag=float(mag), de_err=float(err[j]),
                                 depth=float(abs(R0[j, ks[j], ms[j]])), env=bool(is_env[j]),
                                 **{k: float(v[j]) for k, v in D.items()}))
        print(f"  {s0 + B} / {len(pairs)}", flush=True)

    col = lambda k: np.array([r[k] for r in rows])
    m_, kind, err = col("m"), col("kind"), col("de_err")

    def tau(d, dstar, sel):
        """距離 d が D* を超える割合が 50% になる d·e の誤差（誤差の分位でビン分けして線形補間）"""
        if sel.sum() < 30:
            return np.nan
        e, v = err[sel], d[sel]
        edges = np.unique(np.quantile(e, np.linspace(0, 1, 9)))
        xs, ps = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (e >= lo) & (e <= hi)
            if s.sum() >= 5:
                xs.append(np.median(e[s]))
                ps.append(np.mean(v[s] > dstar))
        xs, ps = np.array(xs), np.maximum.accumulate(np.array(ps))
        if len(xs) < 2 or ps[0] >= 0.5:
            return float(xs[0]) if len(xs) else np.nan     # 最小の誤差ですでに半数が超える → 上限として最小の誤差
        if ps[-1] < 0.5:
            return np.inf
        i = np.argmax(ps >= 0.5)
        return float(np.interp(0.5, ps[i - 1:i + 1], xs[i - 1:i + 1]))

    print("\n==== 変調先ごとの許容誤差 τ_m（d·e の平均絶対差。距離が D*（attack と release の両方の中央値）を"
          "超える割合が 50% になる誤差） ====")
    for k in [k for k in rows[0] if k in Dstar]:
        d = col(k)
        print(f"\n  [{k}]  D* = {Dstar[k]['both']:.4g}")
        print(f"  {'変調先':10s}{'件数':>6s}{'τ（深さの誤差）':>16s}{'τ（ENV 形状）':>14s}{'τ（深さ＋ENV）':>16s}"
              f"{'LFO 位相 0.05 で超える割合':>26s}")
        for m in range(M):
            sm = m_ == m
            t_dep = tau(d, Dstar[k]["both"], sm & (kind == "depth"))
            t_env = tau(d, Dstar[k]["both"], sm & (kind == "env"))
            t_all = tau(d, Dstar[k]["both"], sm & np.isin(kind, ["depth", "env"]))
            sp_ = sm & (kind == "lfo_phase") & np.isclose(col("mag"), 0.05)
            ph = f"{100 * np.mean(d[sp_] > Dstar[k]['both']):.0f}%（{sp_.sum()}）" if sp_.any() else "-"
            print(f"  {NAMES[m]:10s}{sm.sum():6d}{t_dep:16.3f}{t_env:14.3f}{t_all:16.3f}{ph:>26s}")

    np.savez(args.out, **{k: col(k) for k in rows[0]},
             **{f"Dstar_{k}": np.array([v["attack"], v["release"], v["both"]]) for k, v in Dstar.items()})


if __name__ == "__main__":
    main()
