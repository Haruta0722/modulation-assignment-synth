"""
diag_shape.py — 連続パラメータ（形状）が学習できない原因の切り分け
================================================================================
現状: 対角知覚計量（条件数 2e7）を入れても shape_ok≈0.09 / depth_ok≈0.20 が動かない。
      → 損失の形の問題ではない。真因の候補は
        (1) 情報抽出が弱い（log-mel 64bin / hop256 → 94フレームでは形状が読めない）
        (2) 分離機構がない（複数変調が重畳した音から特定列の軌跡を分離できない）
        (3) 出力表現（スカラー回帰が多峰分布を平均で潰す）

実験デザイン: タスクを段階的に易しくし、どこで解けるようになるかを見る。
  --task single : 変調先1列・ソース1本だけの音から、その形状を当てる
                  → 解けなければ (1)、解ければ (2) が濃厚
  --task oracle : 通常データだが「どの列が動いているか」を入力に与える
                  → 検出と分離を免除。形状推定だけの純粋な能力を測る
  --task full   : 通常タスク（比較用ベースライン）

  --hop / --mel で時間・周波数分解能を変えられる（診断3）。

評価: 予測ソース信号 e(n) と真の e(n) の平均絶対誤差（信号空間、[0,1] スケール）。
      shape_ok = MAE ≤ 0.1 の割合（train.py の shape_ok_on_hit と同じ基準）。

使い方:
  python diag_shape.py --task single --steps 3000
  python diag_shape.py --task single --steps 3000 --hop 128 --mel 128
  python diag_shape.py --task oracle --steps 3000
  python diag_shape.py --task full   --steps 3000
"""
import argparse, json, os
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast

K, M = ds.K_ENVS, ds.M_DEST
F0_LO = 65.4
R_MAX = 0.6


# ---------------------------------------------------------------- データ生成
def make_batch(rng, n, task, sr, N, t_off, fix_dest=-1):
    """診断用データ。task=single なら 1列・1ソースのみ"""
    theta = rng.uniform(0.2, 0.6, (n, M)).astype(np.float32)
    theta[:, 3] = np.minimum(theta[:, 2] + 0.1, 0.95)
    R = np.zeros((n, K, M), np.float32)
    styp = (rng.random((n, K)) < 0.5).astype(np.float32)
    src = np.zeros((n, K, 5), np.float32)
    # ENV パラメータ
    u = rng.uniform(0.1, 0.9, (n, K, 3)).astype(np.float32)
    a = np.maximum(u[..., 0] * t_off * 0.5, 1e-3)
    h = u[..., 1] * (t_off - a) * 0.5
    d = np.maximum(u[..., 2] * (t_off - a - h), 1e-3)
    s = rng.uniform(0.1, 0.9, (n, K)).astype(np.float32)
    r = rng.uniform(0.05, 0.4, (n, K)).astype(np.float32)
    env_p = np.stack([a, h, d, s, r], -1)
    # LFO パラメータ
    rate = (0.5 * (20.0 ** rng.uniform(0, 1, (n, K)))).astype(np.float32)
    ph = rng.uniform(0, 1, (n, K)).astype(np.float32)
    mo = rng.uniform(0, 1, (n, K)).astype(np.float32)
    z0 = np.zeros((n, K), np.float32)
    lfo_p = np.stack([rate, ph, mo, z0, z0], -1)
    src = np.where(styp[..., None] == 1.0, env_p, lfo_p).astype(np.float32)

    dest = np.zeros((n,), np.int64)
    if task == "single":
        # 1列・1ソース。変調先は「王道」から選ぶ（level1, cutoff, rate, depth）
        cand = np.array([fix_dest]) if fix_dest >= 0 else np.array([0, 4, 7, 8])
        dest = cand[rng.integers(0, len(cand), n)]
        for i in range(n):
            R[i, 0, dest[i]] = rng.uniform(0.3, 0.8) * (1 if rng.random() < .5 else -1)
            theta[i, dest[i]] = np.clip(theta[i, dest[i]], 0.15, 1 - abs(R[i, 0, dest[i]]))
    else:
        # 通常: 3〜5本
        for i in range(n):
            J = rng.integers(3, 6)
            cols = rng.choice(M, J, replace=False)
            cols[0] = rng.integers(0, 2)          # D4: level
            for m in cols:
                k = rng.integers(0, K)
                dep = rng.uniform(0.2, 1.0) * (1 if (m < 2 or rng.random() < .5) else -1)
                theta[i, m] = np.clip(theta[i, m], 0.1, max(0.11, 1 - abs(dep)))
                R[i, k, m] = dep
        dest = np.argmax(np.abs(R).sum(1), axis=1)
    osc = rng.uniform(0, 1, (n, 2)).astype(np.float32)
    gmk = rng.uniform(0, 1, (n,)).astype(np.float32)
    f0 = (440.0 * 2 ** ((np.array([36, 48, 60, 72, 84])[
        rng.integers(0, 5, n)] - 69) / 12)).astype(np.float32)
    return theta, osc, gmk, src, styp, R, f0, dest


# ---------------------------------------------------------------- モデル
class ShapeNet(tf.keras.Model):
    """1本のソース信号 e(n) を直接（フレーム列として）予測する診断用モデル。
    パラメータ回帰を経由せず信号を出すので、『情報が音にあるか』を純粋に測れる。"""

    def __init__(self, sr, n_frames, hop=256, mel=64, oracle=False):
        super().__init__()
        self.sr, self.hop, self.n_mel = sr, hop, mel
        self.oracle = oracle
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(mel, 513, sr, 30., 7800.)
        self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                      for c, st in ((32, (1, 2)), (64, (1, 2)), (96, (1, 2)), (128, (1, 2)))]
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=True))
        self.head = tf.keras.layers.Dense(1)
        self.n_frames = n_frames

    def call(self, x, f0, dest_oh=None):
        S = tf.abs(tf.signal.stft(x, 1024, self.hop, fft_length=1024))
        h = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)[..., None]
        for c in self.convs:
            h = c(h)
        B, T = tf.shape(h)[0], tf.shape(h)[1]
        h = tf.reshape(h, [B, T, h.shape[2] * h.shape[3]])
        f0n = tf.math.log(f0 / F0_LO)[:, None] / np.log(16.0)
        cond = f0n[:, None, :] * tf.ones([1, T, 1])
        if self.oracle and dest_oh is not None:      # どの列が動いているかを与える
            cond = tf.concat([cond, dest_oh[:, None, :] * tf.ones([1, T, 1])], -1)
        h = tf.concat([h, cond], -1)
        h = self.bigru(h)
        return tf.sigmoid(self.head(h))[..., 0]       # [B,T] ∈ [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["single", "oracle", "full"], default="single")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--mel", type=int, default=64)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--toff", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dest", type=int, default=-1,
                    help="変調先を1つに固定（0-16）。-1 なら王道4列からランダム")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    SR, N, T_OFF = args.sr, int(args.sr * args.T), args.toff
    n_frames = 1 + (N - 1024) // args.hop
    fmax = F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    model = ShapeNet(SR, n_frames, args.hop, args.mel, oracle=(args.task == "oracle"))
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function(reduce_retracing=True)
    def synth(theta, osc, gmk, src, R, f0, styp):
        return synth_fast(theta, osc, gmk, src, R, f0, T_OFF, N, SR,
                          NH[0], NH[1], src_type=styp)

    def target_env(src, styp, dest, R):
        """その列を動かしているソースの信号 e(n) をフレームにダウンサンプル"""
        e = ds.sources(tf.constant(src), tf.constant(styp), T_OFF, N, SR).numpy()
        k = np.argmax(np.abs(R[np.arange(len(dest)), :, dest]), axis=1)
        e_sel = e[np.arange(len(dest)), k]                       # [n,N]
        idx = np.minimum((np.arange(n_frames) * args.hop + 512), N - 1)
        return e_sel[:, idx].astype(np.float32)                  # [n,T]

    @tf.function(reduce_retracing=True)
    def step(x, f0, doh, y):
        with tf.GradientTape() as tp:
            p = model(x, f0, doh)
            loss = tf.reduce_mean(tf.abs(p - y))
        g = tp.gradient(loss, model.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    dn = ds.dest_specs(SR)[args.dest][0] if args.dest >= 0 else "王道4列"
    print(f"task={args.task}  hop={args.hop} mel={args.mel} frames={n_frames} dest={dn}")
    hist = []
    for it in range(args.steps + 1):
        th, osc, gmk, src, styp, R, f0, dest = make_batch(
            rng, args.batch, args.task, SR, N, T_OFF, args.dest)
        x = synth(*[tf.constant(v) for v in (th, osc, gmk, src, R, f0, styp)])
        y = target_env(src, styp, dest, R)
        doh = tf.constant(np.eye(M, dtype=np.float32)[dest])
        l = step(x, tf.constant(f0), doh, tf.constant(y))
        hist.append(float(l))
        if it % 250 == 0:
            print(f"  step {it:5d}  shape MAE = {np.mean(hist[-50:]):.4f}", flush=True)

    # --- 評価: shape_ok（MAE ≤ 0.1 の割合）---
    oks, maes = [], []
    for _ in range(20):
        th, osc, gmk, src, styp, R, f0, dest = make_batch(
            rng, args.batch, args.task, SR, N, T_OFF, args.dest)
        x = synth(*[tf.constant(v) for v in (th, osc, gmk, src, R, f0, styp)])
        y = target_env(src, styp, dest, R)
        doh = tf.constant(np.eye(M, dtype=np.float32)[dest])
        p = model(x, tf.constant(f0), doh).numpy()
        e = np.abs(p - y).mean(1)
        maes += list(e); oks += list(e <= 0.1)
    print(f"\n== task={args.task} hop={args.hop} mel={args.mel} dest={dn} ==")
    print(f"  shape MAE   = {np.mean(maes):.4f}")
    print(f"  shape_ok    = {np.mean(oks):.3f}   (現行モデルの shape_ok_on_hit ≈ 0.089)")
    print("  判定: shape_ok が高い → 情報は音にある（分離/出力表現の問題）")
    print("        shape_ok が低い → 知覚表現の情報抽出が不足（候補1）")


if __name__ == "__main__":
    main()
