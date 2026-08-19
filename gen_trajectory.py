"""
gen_trajectory.py — 生成的アプローチ 第一段階
================================================================================
目的:
  診断で「変調先ごとに軌跡の復元可能性が 6.3 倍違う」ことが判明した
  （cutoff 0.713 〜 res_q 0.113）。決定論的回帰はこの差を 1 つの MAE に潰すが、
  **なぜ読めないのか**（真値から系統的にずれるのか／複数の軌跡が同じ音を
  説明しうるので平均が中間に落ちるのか）を区別できない。

  そこで条件付き生成モデルで事後分布 p(e | x, dest) を推定し、
  **分布の広がり = 識別可能性の欠如**として直接測る。

手法: 条件付きフローマッチング（Hayes et al. 2025 と同系統。実装が軽く安定）
  x_0 ~ N(0,I), x_1 = 真の軌跡, x_t = (1-t)x_0 + t x_1
  目標速度 v = x_1 - x_0 を v_θ(x_t, t, cond) で回帰
  サンプリング: x_0 から Euler 法で t: 0→1 に積分

測る指標:
  mean_mae   : サンプル平均と真値の MAE（決定論的回帰と比較可能）
  best_mae   : N サンプル中の最良（真値がモードのどれかに含まれるか）
  spread     : サンプル間の標準偏差の平均（= 事後分布の広がり = 縮退の大きさ）
  coverage   : 真値がサンプルの [min,max] に入るフレームの割合
               → spread が大きくても coverage が高ければ「正直な不確実性」、
                 低ければ「単に学習できていない」。両者の区別に必須。

使い方:
  python gen_trajectory.py --dest 4  --steps 4000      # cutoff（読める）
  python gen_trajectory.py --dest 0  --steps 4000      # level1（読めない）
  python gen_trajectory.py --dest 6  --steps 4000      # res_q（最下位）
  for d in $(seq 0 16); do python gen_trajectory.py --dest $d --steps 4000; done
"""
import argparse
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast
from diag_shape import make_batch, F0_LO

K, M = ds.K_ENVS, ds.M_DEST


class FlowNet(tf.keras.Model):
    """条件付き速度場 v_θ(x_t, t, audio) — 軌跡 [B,T] を生成する"""

    def __init__(self, sr, hop=256, mel=64):
        super().__init__()
        self.hop = hop
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(mel, 513, sr, 30., 7800.)
        # 音声エンコーダ（時間軸を保存）
        self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                      for c, st in ((32, (1, 2)), (64, (1, 2)),
                                    (96, (1, 2)), (128, (1, 2)))]
        self.enc_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(96, return_sequences=True))
        # 速度場（x_t と t を条件に結合してから時間方向に処理）
        self.mix = tf.keras.layers.Dense(128, activation="relu")
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(96, return_sequences=True))
        self.out = tf.keras.layers.Dense(1)

    def encode(self, x, f0):
        S = tf.abs(tf.signal.stft(x, 1024, self.hop, fft_length=1024))
        h = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)[..., None]
        for c in self.convs:
            h = c(h)
        B, T = tf.shape(h)[0], tf.shape(h)[1]
        h = tf.reshape(h, [B, T, h.shape[2] * h.shape[3]])
        f0n = tf.math.log(f0 / F0_LO)[:, None] / np.log(16.0)
        h = tf.concat([h, f0n[:, None, :] * tf.ones([1, T, 1])], -1)
        return self.enc_gru(h)                                  # [B,T,192]

    def velocity(self, xt, t, cond):
        """xt:[B,T]  t:[B]  cond:[B,T,C] → v:[B,T]"""
        T = tf.shape(cond)[1]
        tt = t[:, None, None] * tf.ones([1, T, 1])
        # t の周期埋め込み（滑らかな時刻条件付け）
        temb = tf.concat([tf.sin(np.pi * tt), tf.cos(np.pi * tt), tt], -1)
        z = tf.concat([cond, xt[..., None], temb], -1)
        return self.out(self.gru(self.mix(z)))[..., 0]

    def sample(self, cond, n_steps=20, seed=None):
        """x_0 ~ N(0,I) から Euler 積分で軌跡を生成"""
        B, T = tf.shape(cond)[0], tf.shape(cond)[1]
        x = tf.random.normal([B, T], seed=seed)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = tf.fill([B], i * dt)
            x = x + dt * self.velocity(x, t, cond)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=int, default=4, help="変調先（0-16）")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n_samples", type=int, default=16, help="評価時のサンプル数")
    ap.add_argument("--ode_steps", type=int, default=20)
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
    fmax = F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)
    dname = ds.dest_specs(SR)[args.dest][0]

    model = FlowNet(SR, args.hop, args.mel)
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function(reduce_retracing=True)
    def synth(theta, osc, gmk, src, R, f0, styp):
        return synth_fast(theta, osc, gmk, src, R, f0, T_OFF, N, SR,
                          NH[0], NH[1], src_type=styp)

    def target_env(src, styp, dest, R):
        e = ds.sources(tf.constant(src), tf.constant(styp), T_OFF, N, SR).numpy()
        k = np.argmax(np.abs(R[np.arange(len(dest)), :, dest]), axis=1)
        idx = np.minimum(np.arange(n_frames) * args.hop + 512, N - 1)
        return e[np.arange(len(dest)), k][:, idx].astype(np.float32)

    def batch():
        th, osc, gmk, src, styp, R, f0, dest = make_batch(
            rng, args.batch, "single", SR, N, T_OFF, args.dest)
        x = synth(*[tf.constant(v) for v in (th, osc, gmk, src, R, f0, styp)])
        y = target_env(src, styp, dest, R)
        return x, tf.constant(f0), tf.constant(y)

    @tf.function(reduce_retracing=True)
    def step(x, f0, y):
        B = tf.shape(y)[0]
        x0 = tf.random.normal(tf.shape(y))
        t = tf.random.uniform([B])
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * y
        with tf.GradientTape() as tp:
            cond = model.encode(x, f0)
            v = model.velocity(xt, t, cond)
            loss = tf.reduce_mean((v - (y - x0)) ** 2)
        g = tp.gradient(loss, model.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    print(f"dest={dname}  frames={n_frames}  flow matching")
    hist = []
    for it in range(args.steps + 1):
        x, f0, y = batch()
        hist.append(float(step(x, f0, y)))
        if it % 500 == 0:
            print(f"  step {it:5d}  FM loss = {np.mean(hist[-100:]):.4f}", flush=True)

    # ---------------- 評価: 事後分布の広がりと較正 ----------------
    mean_maes, best_maes, spreads, covs = [], [], [], []
    for _ in range(12):
        x, f0, y = batch()
        cond = model.encode(x, f0)
        S = np.stack([model.sample(cond, args.ode_steps).numpy()
                      for _ in range(args.n_samples)], 0)       # [S,B,T]
        yt = y.numpy()
        mean_maes += list(np.abs(S.mean(0) - yt).mean(1))
        best_maes += list(np.abs(S - yt[None]).mean(2).min(0))
        spreads += list(S.std(0).mean(1))
        inside = (yt >= S.min(0)) & (yt <= S.max(0))
        covs += list(inside.mean(1))

    mm, bm = float(np.mean(mean_maes)), float(np.mean(best_maes))
    sp, cv = float(np.mean(spreads)), float(np.mean(covs))
    print(f"\n== dest={dname} (m={args.dest}) ==")
    print(f"  mean_mae  = {mm:.4f}   （サンプル平均 vs 真値。決定論回帰と比較可能）")
    print(f"  best_mae  = {bm:.4f}   （{args.n_samples} サンプル中の最良）")
    print(f"  spread    = {sp:.4f}   ← **事後分布の広がり = 識別可能性の欠如**")
    print(f"  coverage  = {cv:.3f}   （真値がサンプル範囲に入る割合）")
    print(f"  gain      = {mm - bm:+.4f}  （mean−best。正で大きいほど多峰＝解が複数）")
    if bm > mm:
        print("    ※ best > mean は学習不足のサイン（サンプルがノイズ的で"
              "平均の方が真値に近い）。--steps を増やすこと")
    print("\n  読み: spread 小 + coverage 高 → 識別可能（cutoff を期待）")
    print("        spread 大 + coverage 高 → 縮退を正直に表現（level を期待）")
    print("        spread 大 + coverage 低 → 学習不足（要追加学習）")


if __name__ == "__main__":
    main()
