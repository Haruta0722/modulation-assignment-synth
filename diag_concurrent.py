"""
diag_concurrent.py — 複数変調が重畳したときの識別可能性の崩壊を診断
================================================================================
既知:
  - 変調1本のみ: shape_ok は変調先ごとに 0.113〜0.713（6.3倍差）
  - 実タスク（3〜5本）: shape_ok ≈ 0.089（ほぼ全滅）
  - oracle（どの列かを教える）でも 0.100 → 「分離機構の欠如」ではない
  → **重畳した時点で個々の軌跡の痕跡が音から失われている**

本スクリプトが埋める空白: その中間で何が起きているか。

  --mode nway   : 同時変調本数 J=1,2,3,4,5 で対象変調先の軌跡復元を測る
                  → 単調劣化か、ある本数で急落か（崩壊の形）
  --mode pair   : 対象 m と干渉源 m' の全ペアで測る（干渉行列）
                  → どの変調先と共存すると壊れるか。スパイクのクラスタと対応するか
  --mode mask   : 干渉源の深さを 0→0.8 で掃引
                  → 崩壊が深さに比例（マスキング）か閾値的（混同）か

いずれも対象変調先の軌跡のみを予測させる（oracle 設定: 対象列は既知）。
干渉源は対象とは別のソースに割り当てる（同一列への重畳は列スパース制約で禁止）。

使い方:
  python diag_concurrent.py --mode nway --target 4 --steps 2500
  python diag_concurrent.py --mode pair --target 4 --steps 2000
  python diag_concurrent.py --mode mask --target 4 --interferer 5 --steps 2000
"""
import argparse
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast
from diag_shape import ShapeNet, F0_LO

K, M = ds.K_ENVS, ds.M_DEST


def make_conc(rng, n, target, sr, N, t_off, n_mod=1, interferers=None,
              inter_depth=None):
    """対象 target に1本、加えて干渉変調を n_mod-1 本（または指定リスト）刺す"""
    theta = rng.uniform(0.25, 0.55, (n, M)).astype(np.float32)
    theta[:, 3] = np.minimum(theta[:, 2] + 0.1, 0.95)
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

    pool = [m for m in range(M) if m != target]
    for i in range(n):
        # 対象は必ずソース0
        dep = rng.uniform(0.3, 0.8) * (1 if rng.random() < .5 else -1)
        theta[i, target] = np.clip(theta[i, target], 0.15, 1 - abs(dep))
        R[i, 0, target] = dep
        # 干渉源
        if interferers is not None:
            cols = list(interferers)
        else:
            cols = list(rng.choice(pool, max(n_mod - 1, 0), replace=False))
        for j, m in enumerate(cols):
            k = 1 + (j % (K - 1))                    # ソース1..3 に割り当て
            dd = (inter_depth if inter_depth is not None
                  else rng.uniform(0.3, 0.8)) * (1 if rng.random() < .5 else -1)
            if abs(dd) < 1e-6:
                continue
            theta[i, m] = np.clip(theta[i, m], 0.1, max(0.11, 1 - abs(dd)))
            R[i, k, m] = dd
    osc = rng.uniform(0, 1, (n, 2)).astype(np.float32)
    gmk = rng.uniform(0, 1, (n,)).astype(np.float32)
    f0 = (440.0 * 2 ** ((np.array([36, 48, 60, 72, 84])[
        rng.integers(0, 5, n)] - 69) / 12)).astype(np.float32)
    return theta, osc, gmk, src, styp, R, f0


def run_one(args, rng, n_mod=1, interferers=None, inter_depth=None, tag=""):
    """1 条件を学習して shape_ok を返す"""
    SR, N, T_OFF = args.sr, int(args.sr * args.T), args.toff
    n_frames = 1 + (N - 1024) // args.hop
    fmax = F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    tf.random.set_seed(args.seed)
    model = ShapeNet(SR, n_frames, args.hop, args.mel, oracle=False)
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function(reduce_retracing=True)
    def synth(th, osc, gmk, src, R, f0, styp):
        return synth_fast(th, osc, gmk, src, R, f0, T_OFF, N, SR,
                          NH[0], NH[1], src_type=styp)

    idx = np.minimum(np.arange(n_frames) * args.hop + 512, N - 1)

    def batch():
        th, osc, gmk, src, styp, R, f0 = make_conc(
            rng, args.batch, args.target, SR, N, T_OFF, n_mod,
            interferers, inter_depth)
        x = synth(*[tf.constant(v) for v in (th, osc, gmk, src, R, f0, styp)])
        e = ds.sources(tf.constant(src), tf.constant(styp), T_OFF, N, SR).numpy()
        y = e[:, 0][:, idx].astype(np.float32)        # 対象はソース0
        return x, tf.constant(f0), tf.constant(y)

    @tf.function(reduce_retracing=True)
    def step(x, f0, y):
        with tf.GradientTape() as tp:
            loss = tf.reduce_mean(tf.abs(model(x, f0, None) - y))
        g = tp.gradient(loss, model.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    for it in range(args.steps + 1):
        x, f0, y = batch()
        step(x, f0, y)

    oks, maes = [], []
    for _ in range(12):
        x, f0, y = batch()
        e = np.abs(model(x, f0, None).numpy() - y.numpy()).mean(1)
        maes += list(e); oks += list(e <= 0.1)
    return float(np.mean(oks)), float(np.mean(maes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nway", "pair", "mask"], default="nway")
    ap.add_argument("--target", type=int, default=4)
    ap.add_argument("--interferer", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--mel", type=int, default=64)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--toff", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair_list", type=str, default="",
                    help="pair モードで調べる干渉源（既定は代表8列）")
    args = ap.parse_args()
    names = [s[0] for s in ds.dest_specs(args.sr)]
    tname = names[args.target]

    if args.mode == "nway":
        print(f"[nway] target={tname}: 同時変調本数を変えて軌跡復元を測る")
        print(f"{'J':>3s} {'shape_ok':>9s} {'MAE':>7s}")
        for J in (1, 2, 3, 4, 5):
            rng = np.random.default_rng(args.seed)
            ok, mae = run_one(args, rng, n_mod=J)
            print(f"{J:3d} {ok:9.3f} {mae:7.4f}", flush=True)

    elif args.mode == "pair":
        cand = ([int(v) for v in args.pair_list.split(",")] if args.pair_list
                else [0, 5, 6, 8, 10, 13, 15, 16])
        cand = [c for c in cand if c != args.target]
        rng = np.random.default_rng(args.seed)
        ok0, mae0 = run_one(args, rng, n_mod=1)
        print(f"[pair] target={tname}  単独 shape_ok={ok0:.3f}")
        print(f"{'干渉源':10s} {'shape_ok':>9s} {'低下':>7s}")
        for c in cand:
            rng = np.random.default_rng(args.seed)
            ok, _ = run_one(args, rng, interferers=[c])
            print(f"{names[c]:10s} {ok:9.3f} {ok - ok0:+7.3f}", flush=True)

    else:  # mask
        iname = names[args.interferer]
        print(f"[mask] target={tname}  干渉源={iname}  深さ掃引")
        print(f"{'depth':>6s} {'shape_ok':>9s} {'MAE':>7s}")
        for dep in (0.0, 0.2, 0.4, 0.6, 0.8):
            rng = np.random.default_rng(args.seed)
            ok, mae = run_one(args, rng, interferers=[args.interferer],
                              inter_depth=dep)
            print(f"{dep:6.1f} {ok:9.3f} {mae:7.4f}", flush=True)
        print("\n  読み: 深さに比例して低下 → マスキング（痕跡が覆い隠される）")
        print("        ある深さで急落      → 混同（両者が区別できなくなる）")


if __name__ == "__main__":
    main()
