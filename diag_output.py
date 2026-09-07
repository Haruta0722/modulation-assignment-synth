"""
diag_output.py — 出力表現がボトルネックかを検証する
================================================================================
これまでに棄却された仮説:
  分離機構の欠如 / 知覚表現の不足 / 損失の形 / 重畳による情報損失 / ソース帰属

残る差分: 診断系は**自由なフレーム列**（90次元）を出力するが、
          実タスクは **ADHSR 5パラメータ or LFO 3パラメータ**に押し込む。

  診断系  shape_ok ≈ 0.6〜0.8（cutoff, J=1〜5）
  実タスク shape_ok ≈ 0.089

本スクリプトは同一データ・同一エンコーダで**出力表現だけを変えて**比較する:

  --out free   : フレーム列を直接出力（診断系と同じ。上限の参照値）
  --out param  : ADHSR/LFO パラメータを出力 → 区分линейで信号を構成して評価
                 （実タスクと同じ。型は Gumbel で推定）
  --out param_oracle : パラメータ出力だが**型を教える**
                 （型誤りの寄与を分離する）

param が free より大きく落ちれば、出力表現がボトルネック。
param_oracle が param より高ければ、その差が「型推定の失敗」の寄与。

使い方:
  python diag_output.py --out free  --target 4 --steps 3000
  python diag_output.py --out param --target 4 --steps 3000
  python diag_output.py --out param_oracle --target 4 --steps 3000
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


class OutNet(tf.keras.Model):
    """共通エンコーダ + 出力表現を切り替えられるヘッド"""

    def __init__(self, sr, t_off, n_frames, hop=256, mel=64, mode="free"):
        super().__init__()
        self.hop, self.mode, self.t_off = hop, mode, t_off
        self.sr, self.N = sr, None
        self.n_frames = n_frames
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(mel, 513, sr, 30., 7800.)
        self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                      for c, st in ((32, (1, 2)), (64, (1, 2)),
                                    (96, (1, 2)), (128, (1, 2)))]
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=True))
        if mode == "free":
            self.head = tf.keras.layers.Dense(1)          # フレームごと
        else:
            self.pool = tf.keras.layers.Dense(256, activation="relu")
            self.h_env = tf.keras.layers.Dense(5)         # a,h,d,s,r
            self.h_lfo = tf.keras.layers.Dense(4)         # rate, cos, sin, morph
            self.h_type = tf.keras.layers.Dense(2)

    def encode(self, x, f0):
        S = tf.abs(tf.signal.stft(x, 1024, self.hop, fft_length=1024))
        h = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)[..., None]
        for c in self.convs:
            h = c(h)
        B, T = tf.shape(h)[0], tf.shape(h)[1]
        h = tf.reshape(h, [B, T, h.shape[2] * h.shape[3]])
        f0n = tf.math.log(f0 / F0_LO)[:, None] / np.log(16.0)
        h = tf.concat([h, f0n[:, None, :] * tf.ones([1, T, 1])], -1)
        return self.bigru(h)                                # [B,T,256]

    def call(self, x, f0, frame_idx, type_oracle=None, temp=0.5):
        h = self.encode(x, f0)
        if self.mode == "free":
            return tf.sigmoid(self.head(h))[..., 0], None
        g = self.pool(tf.concat([tf.reduce_mean(h, 1),
                                 tf.reduce_max(h, 1)], -1))
        # --- ADHSR（割合パラメータ化。実タスクと同一の構造制約）---
        ue = tf.sigmoid(self.h_env(g))
        a = tf.maximum(ue[:, 0] * self.t_off, 1e-3)
        hh = ue[:, 1] * (self.t_off - a)
        dd = tf.maximum(ue[:, 2] * (self.t_off - a - hh), 1e-3)
        s = ue[:, 3]
        r = ue[:, 4] * R_MAX + 1e-3
        p_env = tf.stack([a, hh, dd, s, r], -1)[:, None, :]      # [B,1,5]
        # --- LFO（rate 対数、位相は円環）---
        ul = self.h_lfo(g)
        rate = 0.5 * tf.pow(20.0, tf.sigmoid(ul[:, 0]))
        phase = tf.math.floormod(tf.atan2(ul[:, 2], ul[:, 1]) / (2 * np.pi), 1.0)
        morph = tf.sigmoid(ul[:, 3])
        z = tf.zeros_like(rate)
        p_lfo = tf.stack([rate, phase, morph, z, z], -1)[:, None, :]
        # --- 型 ---
        tl = self.h_type(g)
        if type_oracle is not None:
            w = type_oracle[:, None]                              # 1=ENV
        else:
            gum = -tf.math.log(-tf.math.log(
                tf.random.uniform(tf.shape(tl), 1e-9, 1.0)))
            y = tf.nn.softmax((tl + gum) / temp, -1)
            yh = tf.one_hot(tf.argmax(y, -1), 2, dtype=y.dtype)
            w = (tf.stop_gradient(yh - y) + y)[:, 0:1]
        p = w[..., None] * p_env + (1 - w[..., None]) * p_lfo     # [B,1,5]
        styp = tf.concat([w, tf.zeros_like(w)], -1)[:, :1]        # [B,1]
        e = ds.sources(p, styp, self.t_off, self.N, self.sr)[:, 0]  # [B,N]
        return tf.gather(e, frame_idx, axis=1), tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", choices=["free", "param", "param_oracle"],
                    default="param")
    ap.add_argument("--target", type=int, default=4)
    ap.add_argument("--n_mod", type=int, default=1)
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

    model = OutNet(SR, T_OFF, n_frames, args.hop, args.mel,
                   "free" if args.out == "free" else "param")
    model.N = N
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
        t_true = styp[ar, kt].astype(np.float32)              # 1=ENV
        return x, tf.constant(f0), tf.constant(y), tf.constant(t_true)

    use_oracle = (args.out == "param_oracle")

    @tf.function(reduce_retracing=True)
    def step(x, f0, y, t_true):
        with tf.GradientTape() as tp:
            p, tl = model(x, f0, fidx,
                          type_oracle=(t_true if use_oracle else None))
            loss = tf.reduce_mean(tf.abs(p - y))
            if tl is not None and not use_oracle:            # 型は教師ありで補助
                loss += 0.3 * tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.cast(1.0 - t_true, tf.int32), logits=tl))
        g = tp.gradient(loss, model.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    print(f"out={args.out}  target={tname}  n_mod={args.n_mod}  frames={n_frames}")
    hist = []
    for it in range(args.steps + 1):
        x, f0, y, tt = batch()
        hist.append(float(step(x, f0, y, tt)))
        if it % 500 == 0:
            print(f"  step {it:5d}  loss = {np.mean(hist[-100:]):.4f}", flush=True)

    oks, maes, tacc = [], [], []
    for _ in range(12):
        x, f0, y, tt = batch()
        p, tl = model(x, f0, fidx,
                      type_oracle=(tt if use_oracle else None))
        e = np.abs(p.numpy() - y.numpy()).mean(1)
        maes += list(e); oks += list(e <= 0.1)
        if tl is not None:
            tacc += list((tf.argmax(tl, -1).numpy() == (1 - tt.numpy())).astype(float))
    print(f"\n== out={args.out} target={tname} n_mod={args.n_mod} ==")
    print(f"  shape MAE = {np.mean(maes):.4f}")
    print(f"  shape_ok  = {np.mean(oks):.3f}")
    if tacc:
        print(f"  type_acc  = {np.mean(tacc):.3f}")
    print("\n  比較: free（自由なフレーム列）vs param（ADHSR/LFO 経由）")
    print("        param が大きく低い → 出力表現がボトルネック")
    print("        param_oracle > param → その差が型推定失敗の寄与")


if __name__ == "__main__":
    main()
