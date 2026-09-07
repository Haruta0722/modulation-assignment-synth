"""
two_stage.py — 二段構成: 軌跡生成 → パラメータ当てはめ
================================================================================
診断（diag_output.py）の結論:
  同一データ・同一エンコーダで出力表現だけを変えると
    free（フレーム列を直接出力）   shape_ok = 0.729
    param（ADHSR/LFO パラメータ経由） shape_ok = 0.333
  → **パラメータ化そのものがボトルネック**（型推定は type_acc=0.990 で成功している）

設計:
  Stage 1  音 → 軌跡 e(n) をフレーム列として自由に予測（制約なし）
           教師: 真の軌跡（信号空間の L1）
  Stage 2  軌跡 → ADHSR/LFO パラメータを当てはめる
           教師: (a) 真のパラメータ（あれば）, (b) 再構成軌跡と Stage1 出力の一致

  重要な設計判断:
    - Stage 2 の入力は**軌跡のみ**（音を見ない）。これにより
      「音→軌跡」と「軌跡→パラメータ」が分離され、誤差の波及が局所化する
    - Stage 2 は軽量（軌跡は 1 次元系列なので小さな GRU で足る）
    - 学習は同時（end-to-end）だが、Stage 1 の教師が直接あるため
      パラメータ化の誤差が Stage 1 に逆流しない（stop_gradient で遮断可）

比較対象:
  --mode free    : Stage1 のみ（上限の参照値）
  --mode param   : 単段パラメータ出力（diag_output の param 相当。下限の参照値）
  --mode two     : 二段構成（本手法）
  --detach       : Stage2 の勾配を Stage1 に流さない（誤差波及の遮断）

使い方:
  python two_stage.py --mode two --target 4 --steps 3000
  python two_stage.py --mode two --target 4 --steps 3000 --detach
  python two_stage.py --mode two --target 4 --n_mod 3 --steps 3000
"""
import argparse
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast
from diag_concurrent import make_conc
from diag_shape import F0_LO

K, M = ds.K_ENVS, ds.M_DEST
R_MAX = 0.6


class Stage1(tf.keras.Model):
    """音 → 軌跡（フレーム列）"""

    def __init__(self, sr, hop=256, mel=64):
        super().__init__()
        self.hop = hop
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(mel, 513, sr, 30., 7800.)
        self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                      for c, st in ((32, (1, 2)), (64, (1, 2)),
                                    (96, (1, 2)), (128, (1, 2)))]
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=True))
        self.head = tf.keras.layers.Dense(1)

    def call(self, x, f0):
        S = tf.abs(tf.signal.stft(x, 1024, self.hop, fft_length=1024))
        h = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)[..., None]
        for c in self.convs:
            h = c(h)
        B, T = tf.shape(h)[0], tf.shape(h)[1]
        h = tf.reshape(h, [B, T, h.shape[2] * h.shape[3]])
        f0n = tf.math.log(f0 / F0_LO)[:, None] / np.log(16.0)
        h = tf.concat([h, f0n[:, None, :] * tf.ones([1, T, 1])], -1)
        return tf.sigmoid(self.head(self.bigru(h)))[..., 0]     # [B,T]


class Stage2(tf.keras.Model):
    """軌跡 → ADHSR/LFO パラメータ（音は見ない）"""

    def __init__(self, t_off):
        super().__init__()
        self.t_off = t_off
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(64, return_sequences=True))
        self.pool = tf.keras.layers.Dense(128, activation="relu")
        self.h_env = tf.keras.layers.Dense(5)
        self.h_lfo = tf.keras.layers.Dense(4)
        self.h_type = tf.keras.layers.Dense(2)

    def call(self, traj, type_oracle=None, temp=0.5):
        h = self.gru(traj[..., None])
        g = self.pool(tf.concat([tf.reduce_mean(h, 1), tf.reduce_max(h, 1)], -1))
        ue = tf.sigmoid(self.h_env(g))
        a = tf.maximum(ue[:, 0] * self.t_off, 1e-3)
        hh = ue[:, 1] * (self.t_off - a)
        dd = tf.maximum(ue[:, 2] * (self.t_off - a - hh), 1e-3)
        p_env = tf.stack([a, hh, dd, ue[:, 3], ue[:, 4] * R_MAX + 1e-3], -1)
        ul = self.h_lfo(g)
        rate = 0.5 * tf.pow(20.0, tf.sigmoid(ul[:, 0]))
        phase = tf.math.floormod(tf.atan2(ul[:, 2], ul[:, 1]) / (2 * np.pi), 1.0)
        z = tf.zeros_like(rate)
        p_lfo = tf.stack([rate, phase, tf.sigmoid(ul[:, 3]), z, z], -1)
        tl = self.h_type(g)
        if type_oracle is not None:
            w = type_oracle[:, None]
        else:
            gum = -tf.math.log(-tf.math.log(
                tf.random.uniform(tf.shape(tl), 1e-9, 1.0)))
            y = tf.nn.softmax((tl + gum) / temp, -1)
            yh = tf.one_hot(tf.argmax(y, -1), 2, dtype=y.dtype)
            w = (tf.stop_gradient(yh - y) + y)[:, 0:1]
        p = w[..., None] * p_env[:, None] + (1 - w[..., None]) * p_lfo[:, None]
        styp = tf.concat([w, tf.zeros_like(w)], -1)[:, :1]
        return p, styp, tl                                  # [B,1,5], [B,1], [B,2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["free", "param", "two"], default="two")
    ap.add_argument("--target", type=int, default=4)
    ap.add_argument("--n_mod", type=int, default=1)
    ap.add_argument("--detach", action="store_true",
                    help="Stage2 の勾配を Stage1 に流さない")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--mel", type=int, default=64)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--toff", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    SR, N, T_OFF = args.sr, int(args.sr * args.T), args.toff
    n_frames = 1 + (N - 1024) // args.hop
    fidx = tf.constant(np.minimum(np.arange(n_frames) * args.hop + 512, N - 1))
    fmax = F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)
    tname = ds.dest_specs(SR)[args.target][0]

    s1 = Stage1(SR, args.hop, args.mel)
    s2 = Stage2(T_OFF) if args.mode != "free" else None
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function(reduce_retracing=True)
    def synth(th, osc, gmk, src, R, f0, styp):
        return synth_fast(th, osc, gmk, src, R, f0, T_OFF, N, SR,
                          NH[0], NH[1], src_type=styp)

    def batch():
        th, osc, gmk, src, styp, R, f0, kt = make_conc(
            rng, args.batch, args.target, SR, N, T_OFF, args.n_mod)
        x = synth(*[tf.constant(v) for v in (th, osc, gmk, src, R, f0, styp)])
        e = ds.sources(tf.constant(src), tf.constant(styp), T_OFF, N, SR).numpy()
        ar = np.arange(len(kt))
        y = e[ar, kt][:, fidx.numpy()].astype(np.float32)
        return (x, tf.constant(f0), tf.constant(y),
                tf.constant(styp[ar, kt].astype(np.float32)),
                tf.constant(src[ar, kt].astype(np.float32)))

    def reconstruct(p, st):
        """パラメータ → 軌跡（フレーム）"""
        e = ds.sources(p, st, T_OFF, N, SR)[:, 0]
        return tf.gather(e, fidx, axis=1)

    @tf.function(reduce_retracing=True)
    def step(x, f0, y, t_true, p_true):
        with tf.GradientTape() as tp:
            traj = s1(x, f0)
            loss = tf.reduce_mean(tf.abs(traj - y))          # Stage1 教師（信号空間）
            if s2 is not None:
                inp = tf.stop_gradient(traj) if args.detach else traj
                p, st, tl = s2(inp)
                rec = reconstruct(p, st)
                # Stage2 教師: 真の軌跡への一致（＋型 CE）
                loss += tf.reduce_mean(tf.abs(rec - y))
                loss += 0.3 * tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.cast(1.0 - t_true, tf.int32), logits=tl))
        vs = s1.trainable_variables + (s2.trainable_variables if s2 else [])
        g = tp.gradient(loss, vs)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, vs))
        return loss

    # --- param モード: Stage1 を経由せず直接パラメータ（比較用）---
    if args.mode == "param":
        s2p = Stage2(T_OFF)
        enc = Stage1(SR, args.hop, args.mel)

        @tf.function(reduce_retracing=True)
        def step_p(x, f0, y, t_true):
            with tf.GradientTape() as tp:
                traj = enc(x, f0)                    # 中間軌跡（教師なし）
                p, st, tl = s2p(traj)
                rec = reconstruct(p, st)
                loss = tf.reduce_mean(tf.abs(rec - y))     # 教師は最終出力のみ
                loss += 0.3 * tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.cast(1.0 - t_true, tf.int32), logits=tl))
            vs = enc.trainable_variables + s2p.trainable_variables
            g = tp.gradient(loss, vs)
            g, _ = tf.clip_by_global_norm(g, 1.0)
            opt.apply_gradients(zip(g, vs))
            return loss

    print(f"mode={args.mode}  target={tname}  n_mod={args.n_mod}  "
          f"detach={args.detach}  frames={n_frames}")
    hist = []
    for it in range(args.steps + 1):
        x, f0, y, tt, pt = batch()
        l = step_p(x, f0, y, tt) if args.mode == "param" else step(x, f0, y, tt, pt)
        hist.append(float(l))
        if it % 500 == 0:
            print(f"  step {it:5d}  loss = {np.mean(hist[-100:]):.4f}", flush=True)

    # --- 評価 ---
    ok_s1, ok_final, mae_s1, mae_final, tacc = [], [], [], [], []
    for _ in range(12):
        x, f0, y, tt, pt = batch()
        yn = y.numpy()
        if args.mode == "param":
            traj = enc(x, f0)
            p, st, tl = s2p(traj)
            rec = reconstruct(p, st).numpy()
        else:
            traj = s1(x, f0)
            if s2 is not None:
                p, st, tl = s2(traj)
                rec = reconstruct(p, st).numpy()
            else:
                rec, tl = None, None
        t1 = np.abs(traj.numpy() - yn).mean(1)
        mae_s1 += list(t1); ok_s1 += list(t1 <= 0.1)
        if rec is not None:
            t2 = np.abs(rec - yn).mean(1)
            mae_final += list(t2); ok_final += list(t2 <= 0.1)
            tacc += list((tf.argmax(tl, -1).numpy() == (1 - tt.numpy())).astype(float))

    print(f"\n== mode={args.mode} target={tname} n_mod={args.n_mod} "
          f"detach={args.detach} ==")
    print(f"  Stage1（軌跡）  shape_ok = {np.mean(ok_s1):.3f}  MAE = {np.mean(mae_s1):.4f}")
    if ok_final:
        print(f"  最終（パラメータ経由） shape_ok = {np.mean(ok_final):.3f}  "
              f"MAE = {np.mean(mae_final):.4f}")
        print(f"  type_acc = {np.mean(tacc):.3f}")
        print(f"  段間の損失 = {np.mean(ok_s1) - np.mean(ok_final):+.3f}")
    print("\n  参照値（diag_output, target=cutoff, J=1）: free 0.729 / param 0.333")


if __name__ == "__main__":
    main()
