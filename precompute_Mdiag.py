"""
precompute_Mdiag.py — 知覚計量 M(θ) の**対角成分**をデータセット平均として事前計算
================================================================================
背景:
  等方 L2（現行の λI 版 param 損失）では連続パラメータが学習できていない
  （desc_rec≈0.01、shape_ok≈0.09、depth_ok≈0.20）。λ_param を10倍にしても
  depth_mae は 0.429→0.413 しか動かず、「重みが足りない」のではなく
  「損失の形が等方的である」ことが原因と判明した。

  フル M(θ) = J^T J は 1件 0.5 分（GX10, jvp_chunk=54）で、テスト5k に 38.5h。
  一方**対角成分** M_ii = ||∂(Φ∘g)/∂θ_i||² だけなら、
    - 接ベクトルを 1 方向に固定して**サンプルをバッチ次元に並べられる**
      （フル M は「1サンプル × P 方向」でバッチが埋まらないのが遅さの主因）
    - パラメータ間の相関（非対角）は捨てるが、
      「音に効く方向の誤差を重く罰する」という PNP の本質は保たれる
  さらに真値分布が同一なら**データセット平均の対角**で共有でき、
  サンプルごとの保存すら不要になる（ベクトル 108 個だけ）。

出力: <data>/Mdiag.npz   d_mean[P], d_median[P], d_std[P], n_samples, jtfs 設定

使い方:
  python precompute_Mdiag.py --data dataset_v7 --split train --n 512 --batch 16
  python precompute_Mdiag.py --data dataset_v7 --split train --n 512 --J 6 --Q 4
"""
import argparse, json, os, time
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast
from jtfs_tf import JTFS
from precompute_M import P, SL, pack, unpack

K, M_DEST = ds.K_ENVS, ds.M_DEST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v7")
    ap.add_argument("--split", type=str, default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=512, help="平均に使うサンプル数")
    ap.add_argument("--batch", type=int, default=16, help="同時に流すサンプル数")
    ap.add_argument("--J", type=int, default=8)
    ap.add_argument("--Q", type=int, default=6)
    ap.add_argument("--J_fr", type=int, default=2)
    ap.add_argument("--Q_fr", type=int, default=2)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.data, "manifest.json")))
    SR, N, T_OFF = man["sr"], man["N"], man["t_off"]
    Pz = dict(np.load(os.path.join(args.data, "params.npz")))
    lo, hi = man["splits"][args.split]
    idx = np.arange(lo, min(lo + args.n, hi))
    n = len(idx)

    theta_all = pack(Pz["theta_stat"][idx], Pz["theta_osc"][idx], Pz["gmk"][idx],
                     Pz["src_params"][idx], Pz["R"][idx])
    f0 = Pz["f0"][idx].astype(np.float32)
    styp = Pz["src_type"][idx].astype(np.float32)
    fmax = 65.4 * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    phi = JTFS(J=args.J, Q=args.Q, J_fr=args.J_fr, Q_fr=args.Q_fr, sr=SR)
    B = args.batch
    eye = np.eye(P, dtype=np.float32)

    @tf.function(reduce_retracing=True)
    def jvp_dir(v, f0b, stb, tangent_row):
        """1 方向 tangent に対する J·t を **バッチ全サンプル同時に** 計算 → [B,D]"""
        tg = tf.tile(tangent_row[None, :], [tf.shape(v)[0], 1])
        with tf.autodiff.ForwardAccumulator(primals=v, tangents=tg) as acc:
            t_, o_, g_, s_, R_ = unpack(v)
            x = synth_fast(t_, o_, g_, s_, R_, f0b, T_OFF, N, SR, NH[0], NH[1],
                           src_type=stb)
            y = phi(x)
        return acc.jvp(y)

    diag = np.zeros((n, P), np.float32)
    t0 = time.time()
    for s0 in range(0, n, B):
        sel = slice(s0, min(s0 + B, n))
        v = tf.constant(theta_all[sel])
        f0b = tf.constant(f0[sel])
        stb = tf.constant(styp[sel])
        for i in range(P):
            j = jvp_dir(v, f0b, stb, tf.constant(eye[i]))       # [b,D]
            diag[sel, i] = tf.reduce_sum(j ** 2, axis=1).numpy()
        done = min(s0 + B, n)
        el = time.time() - t0
        print(f"  [{done}/{n}]  {done/max(el,1e-9)*60:.1f} samples/min  "
              f"ETA {(n-done)/max(done/max(el,1e-9),1e-9)/60:.1f} min", flush=True)

    out = args.out or os.path.join(args.data, "Mdiag.npz")
    # サンプルごとに最大値で正規化してから平均（サンプル間のスケール差を吸収）
    dn = diag / np.maximum(diag.max(axis=1, keepdims=True), 1e-12)
    np.savez(out, d_mean=dn.mean(0), d_median=np.median(dn, 0), d_std=dn.std(0),
             d_raw_mean=diag.mean(0), n_samples=n,
             jtfs=dict(J=args.J, Q=args.Q, J_fr=args.J_fr, Q_fr=args.Q_fr))
    print(f"saved: {out}  ({n} samples, {time.time()-t0:.0f}s)")

    # --- 診断: どのパラメータが知覚的に効くか ---
    names = (["stat_" + s[0] for s in ds.dest_specs(SR)] + ["osc_a1", "osc_a2", "gmk"]
             + [f"src{k}_{p}" for k in range(K)
                for p in ("s0", "s1", "s2", "s3", "s4")]
             + [f"R_{k}_{m}" for k in range(K) for m in range(M_DEST)])
    d = np.load(out)["d_mean"]
    order = np.argsort(-d)
    print("\n知覚感度 上位10（音に効く＝誤差を重く罰すべき方向）:")
    for i in order[:10]:
        print(f"  {names[i]:18s} {d[i]:.4e}")
    print("知覚感度 下位10（ほぼ聞こえない＝縮退方向）:")
    for i in order[-10:]:
        print(f"  {names[i]:18s} {d[i]:.4e}")
    print(f"\n条件数（max/min）= {d.max()/max(d.min(),1e-12):.2e}  "
          f"→ 等方 L2 がいかに不適切かの指標")


if __name__ == "__main__":
    main()
