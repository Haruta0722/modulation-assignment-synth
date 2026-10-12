r"""pitch_check.py — 知覚距離（JTFS 相対）が音程の弁別閾とも整合するかを確かめる

  python pitch_check.py --ref calibrate_volume_s08.npz --n 300

音量の attack / release で決めた聞き分けの限界 D* が、別の次元（音程）の弁別閾でも同じ距離になるかを見る。
  global : fin1・fin2（2 つのオシレータの音程の微調整、±100 セント）の静的な値を同じだけずらす → 音全体の音程のずれ
  detune : fin1 だけずらす → 2 つのオシレータの音程差（うなりの速さ）が変わる
ずらす量: 1, 2, 5, 10, 20 セント（上下を半々）。定常音の音程の弁別閾は数セント（純音の 1〜2 kHz で 0.2〜0.3%、約 3.5〜5 セント）。
"""
import argparse, os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

import train as T
from audibility import quantize
from diff_synth_fast import synth_fast
from jtfs_tf import JTFS

CENTS = [1, 2, 5, 10, 20]
FIN1, FIN2 = 2, 3
SPAN = 200.0                                         # fin は −100〜100 セント


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v7")
    ap.add_argument("--ref", default="calibrate_volume_s08.npz")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--antialias", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    ref = np.load(args.ref)
    jnd = np.isclose(np.abs(np.log(ref["factor"])), np.log(1.25)) & (ref["mode"] == "d1")
    Dstar = float(np.median(ref["jtfs_pnp_rel"][jnd]))

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    phi = JTFS(sr=SR, J=8, Q=6, J_fr=2, Q_fr=2, antialias=args.antialias)
    feat = tf.function(phi.call, reduce_retracing=True)

    def synth(b, th):
        f0 = b["f0"].numpy()
        out = np.zeros((len(f0), N), np.float32)
        for f in np.unique(f0):
            s = np.nonzero(f0 == f)[0]
            fmax = float(f) * 2 ** (100 / 1200)
            hs, hq = int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2)
            g = lambda t: tf.gather(tf.constant(t) if isinstance(t, np.ndarray) else t, s)
            out[s] = synth_fast(g(th), g(b["theta_osc"]), g(b["gmk"]), g(b["src_params"]), g(b["R"]),
                                g(b["f0"]), T_OFF, N, SR, hs, hq, src_type=g(b["src_type"])).numpy()
        return quantize(tf.constant(out), data.scale)

    # fin1・fin2 に変調がつながっていない音（静的な値だけで音程が決まる）
    pool = np.concatenate([data.splits["test"], data.splits["val"]])
    z = data.P["z"]
    idx_all = pool[(z[pool, FIN1] < 0) & (z[pool, FIN2] < 0)][:args.n]
    rows = []
    for s0 in range(0, len(idx_all), args.batch):
        idx = idx_all[s0:s0 + args.batch]
        x, b = data.batch(idx)
        th0 = b["theta_stat"].numpy()
        x0 = synth(b, th0)
        F0 = feat(x0)
        n0 = tf.norm(F0, axis=1).numpy()
        for mode in ("global", "detune"):
            for c in CENTS:
                sgn = np.where(rng.random(len(idx)) < 0.5, 1.0, -1.0)
                th = th0.copy()
                th[:, FIN1] = np.clip(th0[:, FIN1] + sgn * c / SPAN, 0, 1)
                if mode == "global":
                    th[:, FIN2] = np.clip(th0[:, FIN2] + sgn * c / SPAN, 0, 1)
                d = tf.norm(feat(synth(b, th)) - F0, axis=1).numpy() / np.maximum(n0, 1e-12)
                rows += [(mode, c, v) for v in d]
        print(f"  {s0 + len(idx)} / {len(idx_all)}", flush=True)

    mode = np.array([r[0] for r in rows]); cent = np.array([r[1] for r in rows]); d = np.array([r[2] for r in rows])
    print(f"\nD*（音量 attack / release ±25%、JTFS 相対）= {Dstar:.4g}   対象の音 {len(idx_all)}")
    print(f"{'ずらし方':8s}" + "".join(f"{str(c) + ' セント':>16s}" for c in CENTS))
    for m in ("global", "detune"):
        line = []
        for c in CENTS:
            s = (mode == m) & (cent == c)
            line.append(f"{np.median(d[s]):.4f}({100 * np.mean(d[s] > Dstar):.0f}%)")
        print(f"{m:8s}" + "".join(f"{t:>16s}" for t in line))
    print("（各セル: 距離の中央値（D* を超える割合））")


if __name__ == "__main__":
    main()
