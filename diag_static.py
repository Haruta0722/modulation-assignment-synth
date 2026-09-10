"""
diag_static.py — 静的パラメータの推定可能性を診断する
================================================================================
背景:
  本タスクの評価で stat_mae = 0.266、gmk_mae = 0.257 と、
  静的パラメータ（変調先のベース値・波形モーフ・makeup）もほぼ当たっていない。
  $\\theta^{stat}$ は [0,1] の値なので、一様分布どうしの期待誤差 0.33 に近い。
  → 連続量の推定失敗は変調の形状・深さだけでなく静的値全般に及んでいる可能性。

  ただし本タスクでは「変調あり・ゲート推定あり・4 ソース同時」という
  複雑な条件下での数字なので、静的パラメータ単体の難しさは分からない。

本スクリプトは条件を単純化して切り分ける:

  --mod none    : 変調を一切かけない（全パラメータ静的）→ 純粋な回帰問題としての上限
  --mod single  : 変調 1 本のみ
  --mod full    : 変調 3〜5 本（本タスクに近い）

  → none で当たるなら「変調が入ると静的値が読めなくなる」
    none でも当たらないなら「そもそも静的パラメータの回帰が難しい」

測定:
  各パラメータごとの MAE（正規化空間 [0,1]）と、
  「一様推測ベースライン 0.333」および「常に平均値 0.5 を答える 0.25」との比較。
  誤差が 0.25 を下回らないパラメータは**実質的に推定できていない**。

使い方:
  python diag_static.py --mod none   --steps 3000
  python diag_static.py --mod single --steps 3000
  python diag_static.py --mod full   --steps 3000
"""
import argparse
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast
from diag_shape import F0_LO

K, M = ds.K_ENVS, ds.M_DEST
NAMES = None


def make_batch(rng, n, mod, sr, N, t_off):
    """静的パラメータを一様サンプルし、指定の本数だけ変調を加える"""
    theta = rng.uniform(0.05, 0.95, (n, M)).astype(np.float32)
    theta[:, 3] = np.minimum(theta[:, 2] + 0.1, 0.95)          # fin 正準順序
    osc = rng.uniform(0.0, 1.0, (n, 2)).astype(np.float32)
    gmk = rng.uniform(0.0, 1.0, (n,)).astype(np.float32)
    R = np.zeros((n, K, M), np.float32)
    styp = (rng.random((n, K)) < 0.5).astype(np.float32)
    u = rng.uniform(0.1, 0.9, (n, K, 3)).astype(np.float32)
    a = np.maximum(u[..., 0] * t_off * 0.5, 1e-3)
    h = u[..., 1] * (t_off - a) * 0.5
    d = np.maximum(u[..., 2] * (t_off - a - h), 1e-3)
    s = rng.uniform(0.1, 0.9, (n, K)).astype(np.float32)
    r = rng.uniform(0.05, 0.4, (n, K)).astype(np.float32)
    env_p = np.stack([a, h, d, s, r], -1)
    rate = (0.5 * (20.0 ** rng.uniform(0, 1, (n, K)))).astype(np.float32)
    ph = rng.uniform(0, 1, (n, K)).astype(np.float32)
    mo = rng.uniform(0, 1, (n, K)).astype(np.float32)
    z0 = np.zeros((n, K), np.float32)
    lfo_p = np.stack([rate, ph, mo, z0, z0], -1)
    src = np.where(styp[..., None] == 1.0, env_p, lfo_p).astype(np.float32)

    n_mod = {"none": 0, "single": 1, "full": None}[mod]
    for i in range(n):
        J = n_mod if n_mod is not None else int(rng.integers(3, 6))
        if J == 0:
            continue
        cols = rng.choice(M, J, replace=False)
        for j, m in enumerate(cols):
            k = j % K
            dep = rng.uniform(0.2, 0.7) * (1 if rng.random() < .5 else -1)
            theta[i, m] = np.clip(theta[i, m], 0.05, max(0.06, 1 - abs(dep)))
            R[i, k, m] = dep
    f0 = (440.0 * 2 ** ((np.array([36, 48, 60, 72, 84])[
        rng.integers(0, 5, n)] - 69) / 12)).astype(np.float32)
    return theta, osc, gmk, src, styp, R, f0


class StaticNet(tf.keras.Model):
    """音 → 静的パラメータ（θ_stat 17 + θ_osc 2 + G_mk 1 = 20 個）"""

    def __init__(self, sr, hop=256, mel=64):
        super().__init__()
        self.hop = hop
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(mel, 513, sr, 30., 7800.)
        self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                      for c, st in ((32, (2, 2)), (64, (2, 2)),
                                    (96, (2, 2)), (128, (2, 2)))]
        self.pool = tf.keras.layers.Dense(256, activation="relu")
        self.head = tf.keras.layers.Dense(M + 3)

    def call(self, x, f0):
        S = tf.abs(tf.signal.stft(x, 1024, self.hop, fft_length=1024))
        h = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)[..., None]
        for c in self.convs:
            h = c(h)
        B = tf.shape(h)[0]
        g = tf.concat([tf.reduce_mean(h, [1, 2]), tf.reduce_max(h, [1, 2])], -1)
        f0n = tf.math.log(f0 / F0_LO)[:, None] / np.log(16.0)
        return tf.sigmoid(self.head(self.pool(tf.concat([g, f0n], -1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", choices=["none", "single", "full"], default="none")
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
    fmax = F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)
    names = [s[0] for s in ds.dest_specs(SR)] + ["osc_a1", "osc_a2", "gmk"]

    model = StaticNet(SR, args.hop, args.mel)
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function(reduce_retracing=True)
    def synth(th, osc, gmk, src, R, f0, styp):
        return synth_fast(th, osc, gmk, src, R, f0, T_OFF, N, SR,
                          NH[0], NH[1], src_type=styp)

    def batch():
        th, osc, gmk, src, styp, R, f0 = make_batch(
            rng, args.batch, args.mod, SR, N, T_OFF)
        x = synth(*[tf.constant(v) for v in (th, osc, gmk, src, R, f0, styp)])
        y = np.concatenate([th, osc, gmk[:, None]], -1).astype(np.float32)
        return x, tf.constant(f0), tf.constant(y)

    @tf.function(reduce_retracing=True)
    def step(x, f0, y):
        with tf.GradientTape() as tp:
            loss = tf.reduce_mean(tf.abs(model(x, f0) - y))
        g = tp.gradient(loss, model.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    print(f"mod={args.mod}  静的パラメータ {M + 3} 個を予測")
    hist = []
    for it in range(args.steps + 1):
        x, f0, y = batch()
        hist.append(float(step(x, f0, y)))
        if it % 500 == 0:
            print(f"  step {it:5d}  MAE = {np.mean(hist[-100:]):.4f}", flush=True)

    errs = []
    for _ in range(16):
        x, f0, y = batch()
        errs.append(np.abs(model(x, f0).numpy() - y.numpy()))
    E = np.concatenate(errs, 0)                                # [n, M+3]
    per = E.mean(0)

    print(f"\n== mod={args.mod}: パラメータ別 MAE ==")
    print("  （参考: 常に 0.5 と答える戦略 = 0.225 / 一様推測 = 0.30）")
    order = np.argsort(per)
    for i in order:
        mark = "✓" if per[i] < 0.15 else ("△" if per[i] < 0.225 else "✗")
        print(f"  {mark} {names[i]:10s} {per[i]:.4f}")
    n_ok = int((per < 0.15).sum())
    print(f"\n  全体 MAE = {per.mean():.4f}")
    print(f"  推定できている（MAE<0.15）: {n_ok}/{M + 3}")
    print(f"  実質推定できていない（MAE≥0.225）: {int((per >= 0.225).sum())}/{M + 3}")
    print("\n  参照: 本タスクの stat_mae = 0.266 / gmk_mae = 0.257")


if __name__ == "__main__":
    main()
