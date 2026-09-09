"""
diag_landscape.py — 軌跡→パラメータ逆問題の地形を測る
================================================================================
背景:
  二段構成の段間損失（Stage1 0.745 → 最終 0.490、TTO 500 反復で 0.594）は
  「非凸性」に由来すると推測してきたが、**確定していない**。
  改善が見られるという事実は、非凸性でも悪条件でも説明できてしまう。

  非凸性 : 谷が複数あり、初期値によって別の谷に落ちる
  悪条件 : 谷は一つだが細長く、勾配降下が谷底に沿って進みにくい
  → **両方同時に成り立つ場合もある**。そのため独立に測る。

測定内容（同一の目標軌跡に対して、複数の初期値から最適化する）:

  【非凸性の指標】
    n_distinct    : 収束点をクラスタリングして得た相異なる解の個数
    param_spread  : 収束点どうしのパラメータ距離（中央値）
    loss_gap      : 最良解と最悪解の再構成誤差の差
    equal_loss_pairs : 「パラメータは離れているのに誤差が同程度」の対の割合
                       → 局所解が実在することの直接証拠

  【悪条件の指標】
    hess_cond     : 各収束点でのヘッセ行列の条件数（λmax/λmin）の中央値
    grad_ratio    : 最急方向と最緩方向の勾配感度比

  判定:
    n_distinct ≈ 1 かつ cond 小   → どちらでもない（別の原因）
    n_distinct > 1                → 非凸（リスタートが有効）
    cond 大                       → 悪条件（スケーリング・前処理が有効）
    両方                          → 両方の対策が必要

使い方:
  python diag_landscape.py --n 64 --inits 8 --iters 500
  python diag_landscape.py --n 64 --inits 8 --iters 500 --env_only
"""
import argparse
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from fit_ceiling import sample_sources

R_MAX = 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="対象サンプル数")
    ap.add_argument("--inits", type=int, default=8, help="1 サンプルあたりの初期値数")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.0, help="初期値の散らばり")
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--toff", type=float, default=1.0)
    ap.add_argument("--env_only", action="store_true")
    ap.add_argument("--tol_param", type=float, default=0.15,
                    help="収束点を同一とみなすパラメータ距離")
    ap.add_argument("--tol_loss", type=float, default=0.02,
                    help="誤差が同程度とみなす閾値")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    SR, N, T_OFF = args.sr, int(args.sr * args.T), args.toff
    n_frames = 1 + (N - 1024) // args.hop
    fidx = tf.constant(np.minimum(np.arange(n_frames) * args.hop + 512, N - 1))
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    src, styp = sample_sources(rng, args.n, T_OFF, args.env_only)
    e_true = ds.sources(tf.constant(src[:, None, :]),
                        tf.constant(styp[:, None]), T_OFF, N, SR)[:, 0]
    Y = tf.gather(e_true, fidx, axis=1).numpy()              # [n,T]
    W = tf.constant(styp)

    def to_params(v_env, v_lfo, w):
        ue = tf.sigmoid(v_env)
        a = tf.maximum(ue[:, 0] * T_OFF, 1e-3)
        h = ue[:, 1] * (T_OFF - a)
        d = tf.maximum(ue[:, 2] * (T_OFF - a - h), 1e-3)
        p_env = tf.stack([a, h, d, ue[:, 3], ue[:, 4] * R_MAX + 1e-3], -1)
        rate = 0.5 * tf.pow(20.0, tf.sigmoid(v_lfo[:, 0]))
        phase = tf.math.floormod(
            tf.atan2(v_lfo[:, 2], v_lfo[:, 1]) / (2 * np.pi), 1.0)
        z = tf.zeros_like(rate)
        p_lfo = tf.stack([rate, phase, tf.sigmoid(v_lfo[:, 3]), z, z], -1)
        ww = w[:, None]
        return (ww * p_env + (1 - ww) * p_lfo)[:, None, :]

    def render(p, w):
        st = tf.stack([w, tf.zeros_like(w)], -1)[:, :1]
        e = ds.sources(p, st, T_OFF, N, SR)[:, 0]
        return tf.gather(e, fidx, axis=1)

    def fit(v0e, v0l, y, w, iters):
        ve, vl = tf.Variable(v0e), tf.Variable(v0l)
        opt = tf.keras.optimizers.Adam(args.lr)

        @tf.function
        def one():
            with tf.GradientTape() as tp:
                per = tf.reduce_mean(
                    tf.abs(render(to_params(ve, vl, w), w) - y), axis=1)
                tot = tf.reduce_sum(per)          # 合計もテープ内で取る
            g = tp.gradient(tot, [ve, vl])
            opt.apply_gradients(zip(g, [ve, vl]))
            return per

        for _ in range(iters):
            l = one()
        return ve.numpy(), vl.numpy(), l.numpy()

    # ---------------- 複数初期値から最適化 ----------------
    print(f"n={args.n} サンプル × {args.inits} 初期値 × {args.iters} 反復")
    all_v, all_loss = [], []
    for k in range(args.inits):
        v0e = rng.normal(0, args.sigma, (args.n, 5)).astype(np.float32)
        v0l = rng.normal(0, args.sigma, (args.n, 4)).astype(np.float32)
        ve, vl, l = fit(tf.constant(v0e), tf.constant(v0l),
                        tf.constant(Y), W, args.iters)
        all_v.append(np.concatenate([ve, vl], -1))          # [n,9]
        all_loss.append(l)
        print(f"  init {k+1}/{args.inits}: MAE {l.mean():.4f}", flush=True)
    V = np.stack(all_v, 1)                                   # [n, inits, 9]
    L = np.stack(all_loss, 1)                                # [n, inits]

    # 収束点を「実際のパラメータ空間」で比較（自由変数は非線形なので）
    P = np.stack([render(to_params(tf.constant(V[:, k, :5]),
                                   tf.constant(V[:, k, 5:]), W), W).numpy()
                  for k in range(args.inits)], 1)            # [n, inits, T] 軌跡
    PA = np.stack([to_params(tf.constant(V[:, k, :5]),
                             tf.constant(V[:, k, 5:]), W).numpy()[:, 0]
                   for k in range(args.inits)], 1)           # [n, inits, 5]
    # パラメータの正規化（時間は t_off、レベルは 1、rate は対数）
    sc = np.array([T_OFF, T_OFF, T_OFF, 1.0, R_MAX], np.float32)
    PAn = PA / sc

    # ---------------- 非凸性 ----------------
    n_dist, spreads, gaps, eq_pairs = [], [], [], []
    for i in range(args.n):
        d = np.linalg.norm(PAn[i][:, None] - PAn[i][None], axis=-1)  # [k,k]
        # 貪欲クラスタリング
        used, clusters = np.zeros(args.inits, bool), 0
        for a in range(args.inits):
            if used[a]:
                continue
            clusters += 1
            used |= (d[a] <= args.tol_param)
        n_dist.append(clusters)
        iu = np.triu_indices(args.inits, 1)
        spreads.append(np.median(d[iu]))
        gaps.append(L[i].max() - L[i].min())
        # 「離れているのに誤差が同程度」= 局所解の直接証拠
        far = d[iu] > args.tol_param
        same = np.abs(L[i][iu[0]] - L[i][iu[1]]) <= args.tol_loss
        eq_pairs.append(float((far & same).mean()))

    # ---------------- 悪条件 ----------------
    # ヤコビアン J（軌跡のパラメータ微分）から Gauss-Newton 近似 J^T J の条件数を測る。
    # 二階微分（ヘッセ）は区分線形シンセでは定義されないため、最小二乗の標準手法を使う。
    conds = []
    for i in range(min(args.n, 16)):
        k = int(np.argmin(L[i]))
        ve = tf.Variable(V[i:i + 1, k, :5])
        vl = tf.Variable(V[i:i + 1, k, 5:])
        w = W[i:i + 1]
        with tf.GradientTape(persistent=True) as tp:
            tp.watch([ve, vl])
            traj = render(to_params(ve, vl, w), w)[0]        # [T]
        Je = tp.jacobian(traj, ve)                            # [T,1,5]
        Jl = tp.jacobian(traj, vl)
        del tp
        if Je is None or Jl is None:
            continue
        J = np.concatenate([Je.numpy()[:, 0, :], Jl.numpy()[:, 0, :]], -1)  # [T,9]
        J = np.nan_to_num(J)
        ev = np.linalg.eigvalsh(J.T @ J)
        ev = np.clip(ev, 0, None)
        if ev.max() <= 0:
            continue
        pos = ev[ev > ev.max() * 1e-12]
        if len(pos) >= 2:
            conds.append(float(pos.max() / pos.min()))

    print(f"\n== 非凸性 ==")
    print(f"  相異なる収束点の数（中央値） = {np.median(n_dist):.1f} / {args.inits} 初期値")
    print(f"  収束点の分布: 1個 {np.mean(np.array(n_dist)==1)*100:.0f}%  "
          f"2個 {np.mean(np.array(n_dist)==2)*100:.0f}%  "
          f"3個以上 {np.mean(np.array(n_dist)>=3)*100:.0f}%")
    print(f"  収束点間のパラメータ距離（中央値） = {np.median(spreads):.3f}")
    print(f"  最良-最悪の誤差差（中央値） = {np.median(gaps):.4f}")
    print(f"  「離れているのに誤差が同程度」の対の割合 = {np.mean(eq_pairs):.3f}")
    print(f"     → 高い（>0.1）ほど局所解が実在する直接証拠")

    print(f"\n== 悪条件 ==")
    if conds:
        print(f"  Gauss-Newton 条件数（中央値） = {np.median(conds):.2e}")
        print(f"  範囲: {np.min(conds):.2e} 〜 {np.max(conds):.2e}")
        print(f"     → 1e4 超なら悪条件（前処理・スケーリングが有効）")
    else:
        print("  計算できず")

    print(f"\n== 判定 ==")
    nc = np.median(n_dist) > 1.2
    ic = bool(conds) and np.median(conds) > 1e4
    print(f"  非凸性: {'あり' if nc else 'なし/弱い'}   悪条件: {'あり' if ic else 'なし/弱い'}")
    if nc and ic:
        print("  → **両方**。リスタート（非凸対策）＋スケーリング（悪条件対策）の併用が必要")
    elif nc:
        print("  → 非凸のみ。多点リスタートが有効")
    elif ic:
        print("  → 悪条件のみ。前処理付き最適化（自然勾配・対角スケーリング）が有効")
    else:
        print("  → どちらでもない。段間損失の原因は別（Stage2 の表現力や学習設定）")


if __name__ == "__main__":
    main()
