"""
fit_ceiling.py — パラメータ化の理論的上限を測る
================================================================================
二段構成の結果（target=cutoff, J=1）:
  Stage1（軌跡）  shape_ok = 0.615
  最終（パラメータ経由） shape_ok = 0.396
  → 段間損失 0.219。軌跡は読めているのにパラメータ化で 1/3 が失われる。

この損失の内訳は 2 つに分かれる:
  (A) **表現力の限界**: 真の軌跡が ADHSR/LFO の形状族で表せない
      → どんなに賢い Stage2 でも到達不能
  (B) **Stage2 の能力不足**: 当てはめが下手（一発回帰の限界）
      → 設計改良で縮められる

本スクリプトは (A) を測る。真の軌跡 e*(n) に対して
**パラメータを直接勾配降下で最適化**し、到達可能な最良の再構成誤差を求める。

  ceiling_ok = 最良フィットでの shape_ok
  段間損失 ≈ (1 - ceiling_ok) 由来  → (A) が主因
  ceiling_ok が高い（1 に近い）      → (B) が主因、Stage2 に伸びしろあり

注意: 真値パラメータが分かっている場合でも、**別の**パラメータの方が
      軌跡を良く近似することがある（フレーム標本化・型混合のため）。
      そのため真値からの初期化と乱数初期化の両方を試し、良い方を採る。

使い方:
  python fit_ceiling.py --n 128 --iters 400
  python fit_ceiling.py --n 128 --iters 400 --env_only    # ENV のみで評価
"""
import argparse
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds

K = ds.K_ENVS
R_MAX = 0.6


def sample_sources(rng, n, t_off, env_only=False):
    """診断系と同じ分布からソースパラメータを生成"""
    styp = (np.ones(n) if env_only else (rng.random(n) < 0.5).astype(np.float32))
    u = rng.uniform(0.1, 0.9, (n, 3)).astype(np.float32)
    a = np.maximum(u[:, 0] * t_off * 0.5, 1e-3)
    h = u[:, 1] * (t_off - a) * 0.5
    d = np.maximum(u[:, 2] * (t_off - a - h), 1e-3)
    s = rng.uniform(0.1, 0.9, n).astype(np.float32)
    r = rng.uniform(0.05, 0.4, n).astype(np.float32)
    env_p = np.stack([a, h, d, s, r], -1)
    rate = (0.5 * (20.0 ** rng.uniform(0, 1, n))).astype(np.float32)
    ph = rng.uniform(0, 1, n).astype(np.float32)
    mo = rng.uniform(0, 1, n).astype(np.float32)
    z = np.zeros(n, np.float32)
    lfo_p = np.stack([rate, ph, mo, z, z], -1)
    src = np.where(styp[:, None] == 1.0, env_p, lfo_p).astype(np.float32)
    return src, styp.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="評価サンプル数")
    ap.add_argument("--iters", type=int, default=400, help="フィットの反復数")
    ap.add_argument("--restarts", type=int, default=3, help="乱数初期化の回数")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--toff", type=float, default=1.0)
    ap.add_argument("--env_only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    SR, N, T_OFF = args.sr, int(args.sr * args.T), args.toff
    n_frames = 1 + (N - 1024) // args.hop
    fidx = tf.constant(np.minimum(np.arange(n_frames) * args.hop + 512, N - 1))
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    src, styp = sample_sources(rng, args.n, T_OFF, args.env_only)
    # 真の軌跡（フレーム標本化）
    e_true = ds.sources(tf.constant(src[:, None, :]),
                        tf.constant(styp[:, None]), T_OFF, N, SR)[:, 0]
    y = tf.gather(e_true, fidx, axis=1).numpy()                # [n,T]

    def build(v_env, v_lfo, w):
        """自由変数 → 制約を満たすパラメータ → 軌跡（フレーム）"""
        ue = tf.sigmoid(v_env)
        a = tf.maximum(ue[:, 0] * T_OFF, 1e-3)
        h = ue[:, 1] * (T_OFF - a)
        d = tf.maximum(ue[:, 2] * (T_OFF - a - h), 1e-3)
        p_env = tf.stack([a, h, d, ue[:, 3], ue[:, 4] * R_MAX + 1e-3], -1)
        ul = v_lfo
        rate = 0.5 * tf.pow(20.0, tf.sigmoid(ul[:, 0]))
        phase = tf.math.floormod(tf.atan2(ul[:, 2], ul[:, 1]) / (2 * np.pi), 1.0)
        z = tf.zeros_like(rate)
        p_lfo = tf.stack([rate, phase, tf.sigmoid(ul[:, 3]), z, z], -1)
        ww = w[:, None]
        p = ww * p_env + (1.0 - ww) * p_lfo
        st = tf.concat([w[:, None], tf.zeros_like(w[:, None])], -1)[:, :1]
        e = ds.sources(p[:, None, :], st, T_OFF, N, SR)[:, 0]
        return tf.gather(e, fidx, axis=1)

    def fit(init_env, init_lfo, w_fixed, tag):
        v_env = tf.Variable(init_env)
        v_lfo = tf.Variable(init_lfo)
        opt = tf.keras.optimizers.Adam(args.lr)

        @tf.function
        def one():
            with tf.GradientTape() as tp:
                rec = build(v_env, v_lfo, w_fixed)
                loss = tf.reduce_mean(tf.abs(rec - y))
            g = tp.gradient(loss, [v_env, v_lfo])
            opt.apply_gradients(zip(g, [v_env, v_lfo]))
            return loss

        for _ in range(args.iters):
            one()
        rec = build(v_env, v_lfo, w_fixed).numpy()
        return np.abs(rec - y).mean(1)                          # [n]

    # 型は真値を与える（型推定は type_acc=0.99 で成功済みなので分離）
    w = tf.constant(styp)

    best = None
    # (1) 真値からの初期化（logit 逆変換）
    def inv_sig(p, eps=1e-4):
        p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))
    a0 = src[:, 0] / T_OFF
    h0 = src[:, 1] / np.maximum(T_OFF - src[:, 0], 1e-4)
    d0 = src[:, 2] / np.maximum(T_OFF - src[:, 0] - src[:, 1], 1e-4)
    ie = np.stack([inv_sig(a0), inv_sig(h0), inv_sig(d0),
                   inv_sig(src[:, 3]), inv_sig(src[:, 4] / R_MAX)], -1).astype(np.float32)
    rate_n = inv_sig(np.log(np.maximum(src[:, 0], 1e-3) / 0.5) / np.log(20.0))
    il = np.stack([rate_n, np.cos(2 * np.pi * src[:, 1]),
                   np.sin(2 * np.pi * src[:, 1]), inv_sig(src[:, 2])], -1).astype(np.float32)
    e_init = fit(ie, il, w, "true-init")
    best = e_init
    print(f"  真値初期化からのフィット: MAE {e_init.mean():.4f}")

    # (2) 乱数初期化を複数回
    for r in range(args.restarts):
        ie_r = rng.normal(0, 1, (args.n, 5)).astype(np.float32)
        il_r = rng.normal(0, 1, (args.n, 4)).astype(np.float32)
        e_r = fit(ie_r, il_r, w, f"rand-{r}")
        print(f"  乱数初期化 {r+1}: MAE {e_r.mean():.4f}")
        best = np.minimum(best, e_r)

    ok = (best <= 0.1).mean()
    print(f"\n== パラメータ化の理論的上限（target 非依存、{args.n} サンプル） ==")
    print(f"  最良フィット MAE   = {best.mean():.4f}")
    print(f"  ceiling shape_ok   = {ok:.3f}")
    print(f"  MAE 分布: 中央値 {np.median(best):.4f}  90%点 {np.percentile(best,90):.4f}")
    # 型別
    if not args.env_only:
        for nm, sel in (("ENV", styp == 1.0), ("LFO", styp == 0.0)):
            if sel.sum():
                print(f"  {nm}: MAE {best[sel].mean():.4f}  "
                      f"shape_ok {(best[sel] <= 0.1).mean():.3f}  (n={int(sel.sum())})")
    print("\n  比較: 二段構成の最終 shape_ok = 0.396 / Stage1 = 0.615")
    print("  読み: ceiling ≈ 0.4 付近 → 段間損失は**表現力の限界**（改良の余地小）")
    print("        ceiling ≈ 0.9 付近 → 段間損失は**Stage2 の能力不足**（改良の余地大）")


if __name__ == "__main__":
    main()
