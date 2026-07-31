"""
analyze_identifiability.py — 局所識別可能性 × 復元精度の層別解析（主張D の完成）
================================================================================
仕様 v8 §9.2-7 / §7 の「$M(\\theta)$ スペクトルは局所識別可能性の測定器」を実証する。

やること:
  1. テスト分割の各サンプルについて、学習済みモデルで復元指標を計算（サンプル単位）
  2. 事前計算した $\\bar M(\\theta)$（precompute_M.py の出力）の固有解析
       - 実効ランク（固有値 > tol の個数）
       - 小固有値方向の数（= 局所的な識別不能方向）
       - 条件数、対数行列式（体積 = 局所的な情報量）
  3. 両者の相関・四分位層別表を出力（識別可能な層ほど復元精度が高いか）
  4. 固有ベクトルをパラメータブロック（stat/osc/gmk/src/R）に射影し、
     「どのブロックが縮退方向に多く含まれるか」を集計
  5. 図: 散布図（識別可能性 vs 復元）、層別バー、ブロック別零空間寄与

使い方:
  python analyze_identifiability.py --data dataset_v7 --ckpt_dir ckpt_nw0.25
  python analyze_identifiability.py --data dataset_v7 --ckpt_dir ckpt_mdiag --out analysis_mdiag
"""
import argparse, json, os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diff_synth_tf as ds
import estimator as E
import train as T

K, M = ds.K_ENVS, ds.M_DEST
P = 108
BLOCKS = {"stat": (0, 17), "osc": (17, 19), "gmk": (19, 20),
          "src": (20, 40), "R": (40, 108)}
TOL = 1e-6                      # 固有値の「ゼロ」判定（M̄ は λmax=1 に正規化済み）


def per_sample_metrics(pred, batch, mapping, sr, t_off, N):
    """サンプル単位の復元指標（train.metrics_np のサンプル分解版）"""
    inv = tf.argsort(mapping, axis=1)
    z = batch["z"].numpy()
    z_re = np.where(z >= 0, tf.gather(inv, tf.maximum(batch["z"], 0), axis=1,
                                      batch_dims=1).numpy(), K)
    g_pred = tf.argmax(pred["gate_logits"], -1, output_type=tf.int32).numpy()
    B = z.shape[0]
    R_hat = E.gather_rows(pred["R"], mapping).numpy()
    Rt = batch["R"].numpy()
    R_true_col, R_hat_col = Rt.sum(1), R_hat.sum(1)
    t_lab = (1.0 - batch["src_type"].numpy()).astype(int)
    t_pred = tf.argmax(E.gather_rows(pred["type_logits"], mapping), -1,
                       output_type=tf.int32).numpy()
    e_true = ds.sources(batch["src_params"], batch["src_type"], t_off, N, sr).numpy()
    sp = E.gather_rows(pred["src_params"], mapping)
    st = E.gather_rows(pred["type_oh"][..., 0], mapping)
    e_pred = ds.sources(sp, st, t_off, N, sr).numpy()

    rows = []
    for b in range(B):
        on_t = z_re[b] < K
        on_p = g_pred[b] < K
        tp = (on_t & on_p & (g_pred[b] == z_re[b])).sum()
        f1 = 2 * tp / max(on_t.sum() + on_p.sum(), 1)
        det_tp = (on_t & on_p).sum()
        f1_det = 2 * det_tp / max(on_t.sum() + on_p.sum(), 1)
        desc = 0
        for m in range(M):
            kt, kp = z_re[b, m], g_pred[b, m]
            if kt >= K or kp >= K:
                continue
            if (t_pred[b, kp] == t_lab[b, kt]
                    and abs(R_hat_col[b, m] - R_true_col[b, m]) <= 0.1
                    and np.abs(e_pred[b, kp] - e_true[b, kt]).mean() <= 0.1):
                desc += 1
        rows.append(dict(F1=f1, F1_detect=f1_det,
                         desc_rec=desc / max(on_t.sum(), 1),
                         n_true=on_t.sum()))
    return rows


def eig_features(Mb):
    """M̄ の固有解析 → 局所識別可能性の特徴量"""
    ev = np.linalg.eigvalsh(Mb)
    ev = np.clip(ev, 0, None)
    pos = ev[ev > TOL]
    return dict(
        eff_rank=len(pos),                                 # 実効ランク（大きいほど識別可能）
        n_null=int((ev <= TOL).sum()),                     # 局所識別不能方向の数
        lam_min_pos=float(pos.min()) if len(pos) else 0.0,
        cond=float(ev.max() / max(pos.min(), 1e-12)) if len(pos) else np.inf,
        logdet=float(np.log(pos + 1e-12).sum()),           # 情報量（体積）
        trace=float(ev.sum()),
    )


def block_nullspace(Mb):
    """小固有値方向がどのパラメータブロックに集中するか（寄与率）"""
    ev, V = np.linalg.eigh(Mb)
    sel = ev <= TOL
    if sel.sum() == 0:
        return {k: 0.0 for k in BLOCKS}
    Vn = V[:, sel]                                          # [P, n_null]
    w = (Vn ** 2).sum(1)                                    # 各座標の零空間寄与
    w = w / max(w.sum(), 1e-12)
    return {k: float(w[a:b].sum()) for k, (a, b) in BLOCKS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v7")
    ap.add_argument("--ckpt_dir", type=str, default="ckpt_nw0.25")
    ap.add_argument("--mdir", type=str, default="", help="既定 <data>/M_test")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--arch", type=str, default="v2")
    ap.add_argument("--out", type=str, default="analysis")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.data, "manifest.json")))
    SR, N, T_OFF = man["sr"], man["N"], man["t_off"]
    mdir = args.mdir or os.path.join(args.data, "M_test")
    mman = json.load(open(os.path.join(mdir, "manifest.json")))
    n_M, idx0 = mman["done"], mman["idx0"]
    Mmm = np.memmap(os.path.join(mdir, "M_matrices.f32.bin"), dtype=np.float32,
                    mode="r", shape=(mman["n"], P, P))
    print(f"M: {n_M} 件（idx {idx0}〜）")

    # --- モデル復元 ---
    data = T.Data(args.data)
    model = E.Estimator(sr=SR, t_off=T_OFF, arch=args.arch)
    model(tf.zeros([1, N]), tf.constant([220.0], tf.float32), temp=1.0, hard=True)
    ck = tf.train.Checkpoint(model=model)
    st = ck.restore(tf.train.latest_checkpoint(args.ckpt_dir)).expect_partial()
    fmax = E.F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    LAM = dict(spec=0.0, param=0.0, assign=1.0, sparse=0.0, none_w=1.0)

    # --- サンプル単位の復元指標 ---
    recs = []
    for s0 in range(0, n_M, args.batch):
        idx = np.arange(idx0 + s0, idx0 + min(s0 + args.batch, n_M))
        x, batch = data.batch(idx)
        pred = model(x, batch["f0"], temp=0.3, hard=True)
        _, _, mapping = E.losses(pred, batch, x, SR, T_OFF, NH, LAM, matching=False)
        recs += per_sample_metrics(pred, batch, mapping, SR, T_OFF, N)
    print(f"復元指標: {len(recs)} サンプル")

    # --- M の固有解析 ---
    feats, blocks = [], []
    for i in range(n_M):
        Mb = np.array(Mmm[i], dtype=np.float64)
        Mb = 0.5 * (Mb + Mb.T)
        feats.append(eig_features(Mb))
        blocks.append(block_nullspace(Mb))

    keys_f = list(feats[0].keys())
    Fm = {k: np.array([f[k] for f in feats]) for k in keys_f}
    Rm = {k: np.array([r[k] for r in recs[:n_M]]) for k in recs[0]}

    # --- 相関 ---
    print("\n== 局所識別可能性 × 復元精度 の相関（Spearman） ==")
    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        ra, rb = ra - ra.mean(), rb - rb.mean()
        return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum() + 1e-12))
    rows = []
    for fk in ("eff_rank", "n_null", "logdet", "lam_min_pos"):
        for rk in ("F1", "F1_detect", "desc_rec"):
            v = np.isfinite(Fm[fk]) & np.isfinite(Rm[rk])
            rho = spearman(Fm[fk][v], Rm[rk][v])
            rows.append((fk, rk, rho))
            print(f"  {fk:12s} vs {rk:10s}: ρ = {rho:+.3f}")

    # --- 四分位層別 ---
    print("\n== 実効ランク四分位による層別（主張D の中心的な表） ==")
    q = np.quantile(Fm["eff_rank"], [0.25, 0.5, 0.75])
    lab = np.digitize(Fm["eff_rank"], q)
    print(f"{'層':12s} {'n':>5s} {'eff_rank':>9s} {'F1':>7s} {'detect':>7s} {'desc':>7s}")
    strat = []
    for g, nm in enumerate(["Q1 (hard)", "Q2", "Q3", "Q4 (easy)"]):
        s = lab == g
        if s.sum() == 0:
            continue
        r = (nm, int(s.sum()), Fm["eff_rank"][s].mean(),
             Rm["F1"][s].mean(), Rm["F1_detect"][s].mean(), Rm["desc_rec"][s].mean())
        strat.append(r)
        print(f"{r[0]:12s} {r[1]:5d} {r[2]:9.1f} {r[3]:7.3f} {r[4]:7.3f} {r[5]:7.3f}")

    # --- ブロック別零空間寄与 ---
    print("\n== 縮退方向のパラメータブロック別寄与（平均） ==")
    bmean = {k: float(np.mean([b[k] for b in blocks])) for k in BLOCKS}
    for k, v in sorted(bmean.items(), key=lambda x: -x[1]):
        print(f"  {k:5s}: {v:.3f}")

    # --- 保存 ---
    np.savez(f"{args.out}.npz", **{f"feat_{k}": v for k, v in Fm.items()},
             **{f"rec_{k}": v for k, v in Rm.items()},
             blocks=np.array([[b[k] for k in BLOCKS] for b in blocks]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].scatter(Fm["eff_rank"], Rm["F1"], s=8, alpha=0.4)
    ax[0].set_xlabel("effective rank of $\\bar M(\\theta)$ (local identifiability)")
    ax[0].set_ylabel("per-sample F1")
    ax[0].set_title(f"identifiability vs recovery (ρ={spearman(Fm['eff_rank'], Rm['F1']):+.2f})")
    ax[0].grid(alpha=.3)
    if strat:
        xs = np.arange(len(strat))
        ax[1].bar(xs - 0.2, [r[3] for r in strat], 0.4, label="F1")
        ax[1].bar(xs + 0.2, [r[4] for r in strat], 0.4, label="detect")
        ax[1].set_xticks(xs); ax[1].set_xticklabels([r[0] for r in strat], fontsize=8)
        ax[1].legend(); ax[1].set_title("stratified by effective rank"); ax[1].grid(alpha=.3)
    ks = list(BLOCKS)
    ax[2].bar(ks, [bmean[k] for k in ks], color="tab:red", alpha=.7)
    ax[2].set_title("null-space contribution by parameter block")
    ax[2].set_ylabel("share"); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{args.out}.png", dpi=130)
    print(f"\nsaved: {args.out}.npz / {args.out}.png")


if __name__ == "__main__":
    main()
