"""
precompute_M.py — PNP の知覚計量 M(θ) 事前計算（仕様 v8 §5.2 / §9.3）
================================================================================
M(θ) = J^T J,  J = ∂(Φ∘g)/∂θ_all |_θ    （Φ = JTFS, g = 微分可能シンセ）

要点:
  - **真値のみに依存**するのでオフライン事前計算できる（学習時は読むだけ）
  - J の列は forward-mode JVP（tf.autodiff.ForwardAccumulator）で作る。
    P=108 本の接ベクトルを**バッチ次元に並べて一度に**流すことで、
    「1件あたり 108 回の順方向」を「1件あたり数回のバッチ順方向」に圧縮する
  - 保存: M_matrices.f32.bin （[n,P,P] memmap, 45KB/件）、M̄ = M/λ_max で正規化
  - manifest に done を持ち **再開可能**（tmux 推奨。SSH 切断でプロセスは死ぬが
    ディスクのファイルは残るので、再実行すれば続きから）

θ_all の並び（P=108, 仕様 §3）:
  [0:17]    θ_stat
  [17:19]   θ_osc
  [19]      G_mk
  [20:40]   src_params (K×5, ENV/LFO スロット共用)
  [40:108]  vec(R) (K×M = 68)

使い方:
  python precompute_M.py --data dataset_v7 --split test              # 必須（5k）
  python precompute_M.py --data dataset_v7 --split train --n 20000   # 予算に応じて
  python precompute_M.py --data dataset_v7 --split test --jvp_chunk 12   # OOM 時
"""
import argparse, json, os, time
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast
from jtfs_tf import JTFS

K, M_DEST = ds.K_ENVS, ds.M_DEST
P = M_DEST + 2 + 1 + 5 * K + K * M_DEST          # 17+2+1+20+68 = 108
SL = dict(stat=(0, 17), osc=(17, 19), gmk=(19, 20), src=(20, 40), R=(40, 108))


def pack(theta, osc, gmk, src, R):
    """真値 → θ_all [B,P]"""
    return np.concatenate([theta, osc, gmk[:, None],
                           src.reshape(len(theta), -1),
                           R.reshape(len(theta), -1)], axis=1).astype(np.float32)


def unpack(v):
    """θ_all [B,P] → シンセ引数（tf）"""
    B = tf.shape(v)[0]
    return (v[:, SL["stat"][0]:SL["stat"][1]],
            v[:, SL["osc"][0]:SL["osc"][1]],
            v[:, SL["gmk"][0]],
            tf.reshape(v[:, SL["src"][0]:SL["src"][1]], [B, K, 5]),
            tf.reshape(v[:, SL["R"][0]:SL["R"][1]], [B, K, M_DEST]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v7")
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=0, help="0=分割全件")
    ap.add_argument("--jvp_chunk", type=int, default=18,
                    help="一度に流す接ベクトル本数（OOM 時に下げる）")
    ap.add_argument("--J", type=int, default=8, help="JTFS オクターブ数（軽量寄り既定）")
    ap.add_argument("--Q", type=int, default=6, help="1次のボイス数")
    ap.add_argument("--J_fr", type=int, default=2)
    ap.add_argument("--Q_fr", type=int, default=2)
    ap.add_argument("--time_check", action="store_true",
                    help="1件だけ計測してコスト見積りを出す（本実行前に必ず）")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    root = args.data
    man = json.load(open(os.path.join(root, "manifest.json")))
    SR, N, T_OFF = man["sr"], man["N"], man["t_off"]
    Pz = dict(np.load(os.path.join(root, "params.npz")))
    lo, hi = man["splits"][args.split]
    idx_all = np.arange(lo, hi)
    if args.n:
        idx_all = idx_all[:args.n]
    n = len(idx_all)

    out_dir = args.out or os.path.join(root, f"M_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "M_matrices.f32.bin")
    man_path = os.path.join(out_dir, "manifest.json")
    mm = np.memmap(bin_path, dtype=np.float32,
                   mode="r+" if os.path.exists(bin_path) else "w+", shape=(n, P, P))
    st = dict(split=args.split, n=n, P=P, idx0=int(idx_all[0]), done=0,
              jtfs=dict(J=args.J, Q=args.Q, J_fr=args.J_fr, Q_fr=args.Q_fr),
              normalized="M / lambda_max")
    if os.path.exists(man_path):
        st["done"] = json.load(open(man_path)).get("done", 0)
        print(f"resuming from done={st['done']}")

    phi = JTFS(J=args.J, Q=args.Q, J_fr=args.J_fr, Q_fr=args.Q_fr, sr=SR)
    theta_all = pack(Pz["theta_stat"][idx_all], Pz["theta_osc"][idx_all],
                     Pz["gmk"][idx_all], Pz["src_params"][idx_all], Pz["R"][idx_all])
    f0_all = Pz["f0"][idx_all].astype(np.float32)
    styp_all = Pz["src_type"][idx_all].astype(np.float32)
    fmax = 65.4 * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    C = args.jvp_chunk
    eye = np.eye(P, dtype=np.float32)

    @tf.function(reduce_retracing=True)
    def jvp_chunk(v0, f0b, stb, tangents):
        """接ベクトル群 tangents:[C,P] に対する J·t を一度に計算 → [C,D]"""
        c = tf.shape(tangents)[0]
        v = tf.tile(v0[None, :], [c, 1])
        with tf.autodiff.ForwardAccumulator(primals=v, tangents=tangents) as acc:
            t_, o_, g_, s_, R_ = unpack(v)
            x = synth_fast(t_, o_, g_, s_, R_, tf.tile(f0b[None], [c]),
                           T_OFF, N, SR, NH[0], NH[1],
                           src_type=tf.tile(stb[None, :], [c, 1]))
            y = phi(x)
        return acc.jvp(y)

    if args.time_check:
        t0 = time.time()
        cols = []
        for s0 in range(0, P, C):
            tg = tf.constant(eye[s0:s0 + C])
            cols.append(jvp_chunk(tf.constant(theta_all[0]), tf.constant(f0_all[0]),
                                  tf.constant(styp_all[0]), tg).numpy())
        dt = time.time() - t0
        print(f"[time_check] 1件 {dt/60:.1f} 分 (jvp_chunk={C}, D={cols[0].shape[1]})")
        for tgt, lbl in ((5000, "test 5k"), (20000, "train 20k"), (90000, "train 90k")):
            for w in (1, 8, 32):
                print(f"  {lbl:10s} x{w:2d} 並列 → {dt*tgt/w/3600:.1f} h")
        print("→ 現実的でなければ: --jvp_chunk を上げる / JTFS を軽くする "
              "(--J 6 --Q 4) / 件数を絞る (--n)")
        return

    lam_path = os.path.join(out_dir, "lambda_max.npy")
    lam_arr = (np.load(lam_path) if os.path.exists(lam_path) else np.zeros(n, np.float64))
    t0 = time.time()
    start_at = st["done"]
    for i in range(st["done"], n):
        cols = []
        for s0 in range(0, P, C):
            tg = tf.constant(eye[s0:s0 + C])
            cols.append(jvp_chunk(tf.constant(theta_all[i]),
                                  tf.constant(f0_all[i]),
                                  tf.constant(styp_all[i]), tg).numpy())
        J = np.concatenate(cols, axis=0)                 # [P, D]
        Mm = J @ J.T                                     # [P,P]
        lmax = float(np.linalg.eigvalsh(Mm)[-1])
        mm[i] = (Mm / max(lmax, 1e-12)).astype(np.float32)
        lam_arr[i] = lmax                       # 正規化前スケール（識別可能性の絶対指標）
        st["done"] = i + 1
        if (i + 1) % 2 == 0 or i + 1 == n:
            mm.flush()
            np.save(lam_path, lam_arr)
            json.dump(st, open(man_path, "w"), indent=1)
            el = time.time() - t0
            n_this_run = i + 1 - start_at
            speed = n_this_run / max(el, 1e-9)          # samples/sec (this run)
            done_now = i + 1
            eta = (n - done_now) / max(speed, 1e-9) / 3600
            print(f"  [{done_now}/{n}]  {60*speed:.2f} samples/min  "
                  f"ETA {eta:.1f} h  (lambda_max={lmax:.3e})", flush=True)
    mm.flush()
    np.save(lam_path, lam_arr)
    json.dump(st, open(man_path, "w"), indent=1)
    print(f"done: {st['done']}/{n} → {bin_path} ({os.path.getsize(bin_path)/1e6:.0f} MB)")

    # --- 健全性チェック（最初の1件） ---
    M0 = np.array(mm[0])
    ev = np.linalg.eigvalsh(M0)
    print("sanity check (sample 0):")
    print(f"  symmetric: {np.allclose(M0, M0.T, atol=1e-5)}")
    print(f"  PSD (min eig >= -1e-5): {ev[0] > -1e-5}  min={ev[0]:.2e} max={ev[-1]:.3f}")
    zero_dirs = int((ev < 1e-6).sum())
    print(f"  near-zero eigendirections: {zero_dirs}/{P}  "
          f"(= 局所的な識別不能方向。§7 の縮退・LFO 未使用スロット等)")


if __name__ == "__main__":
    main()
