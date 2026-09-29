r"""analyze_dump.py — eval_dump.py が保存した .npz から指標を計算し直す。

  python analyze_dump.py dump_two.npz
  python analyze_dump.py dump_single.npz dump_two.npz    # 2 つ渡すと共通集合でも比較

出力:
  A. 既存指標の再現（--eval_only の出力と突き合わせる）
  B. しきい値を振った誤差分布（形状 / 深さ / 両方）
  C. 積 d*e(n) の誤差と、最適スカラー倍で合わせ直した誤差
  D. ソース番号どおりの対応 vs ハンガリアン法の最適対応
  E. 深さの縮み（標準偏差の比・相関）
  F. 2 つ渡したとき: 共通ヒット集合での比較
"""
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment

K, M = 4, 17
CORE8 = [0, 1, 2, 3, 4, 7, 8, 10]          # well-conditioned
CLUSTER = [10, 11, 15, 16]                  # degenerate
WEAK = [5, 6, 9, 12, 13, 14]
TOL = 0.1
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


# ---------------------------------------------------------------- 基本量
class Dump:
    def __init__(self, path):
        d = np.load(path)
        self.path = path
        self.idx = d["idx"]
        self.z_true, self.z_pred = d["z_true"], d["z_pred"]
        self.d_true, self.d_pred = d["d_true"], d["d_pred"]
        self.e_true, self.e_param, self.traj = d["e_true"], d["e_param"], d["traj"]
        self.type_true, self.type_pred = d["type_true"], d["type_pred"]
        self.used = d["used"]
        self.margin = d["gate_margin"] if "gate_margin" in d else None
        self.B = len(self.idx)
        self.true_on = self.z_true >= 0
        self.pred_on = self.z_pred < K
        self.hit = self.true_on & self.pred_on & (self.z_true == self.z_pred)

    def cols(self, mask):
        """mask が立った列を (b, m, k_true, k_pred) の配列で返す"""
        b, m = np.nonzero(mask)
        return b, m, self.z_true[b, m], self.z_pred[b, m]

    def shape_err(self, b, m, k_pred, k_true, signal="param"):
        sig = self.e_param if signal == "param" else self.traj
        return np.abs(sig[b, k_pred] - self.e_true[b, k_true]).mean(-1)

    def depth_err(self, b, m):
        return np.abs(self.d_pred[b, m] - self.d_true[b, m])


def f1(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


# ---------------------------------------------------------------- A. 既存指標
def basic_metrics(D):
    tp = D.hit.sum()
    fp = (D.pred_on & ~D.hit).sum()
    fn = (D.true_on & ~D.hit).sum()
    both = D.pred_on & D.true_on
    print(f"\n[A] 既存指標の再現  ({D.path}, n = {D.B})")
    print(f"  F1              = {f1(tp, fp, fn):.4f}")
    for name, cols in (("core8", CORE8), ("cluster", CLUSTER), ("weak", WEAK)):
        s = np.zeros(M, bool); s[cols] = True
        print(f"  F1_{name:8s}   = "
              f"{f1(D.hit[:, s].sum(), (D.pred_on & ~D.hit)[:, s].sum(), (D.true_on & ~D.hit)[:, s].sum()):.4f}")
    print(f"  F1_detect       = {f1(both.sum(), (D.pred_on & ~D.true_on).sum(), (~D.pred_on & D.true_on).sum()):.4f}")
    print(f"  srcid_acc       = {(D.z_pred[both] == D.z_true[both]).mean():.4f}")
    print(f"  n_pred / n_true = {D.pred_on.sum(1).mean():.2f} / {D.true_on.sum(1).mean():.2f}")

    b, m, kt, kp = D.cols(D.hit)
    sh = D.shape_err(b, m, kp, kt); dp = D.depth_err(b, m)
    ty = D.type_pred[b, kp] == D.type_true[b, kt]
    print(f"  shape_ok_on_hit = {(sh <= TOL).mean():.4f}   (母数 {len(b)})")
    print(f"  depth_ok_on_hit = {(dp <= TOL).mean():.4f}")
    print(f"  typeacc_on_hit  = {ty.mean():.4f}")

    b2, m2, kt2, kp2 = D.cols(both)          # desc_rec は番号を問わない
    good = ((D.shape_err(b2, m2, kp2, kt2) <= TOL)
            & (D.depth_err(b2, m2) <= TOL)
            & (D.type_pred[b2, kp2] == D.type_true[b2, kt2]))
    print(f"  desc_rec_rate   = {good.sum() / max(D.true_on.sum(), 1):.4f}")


# ---------------------------------------------------------------- B. 分布
def sweep(sh, dp, label):
    print(f"\n  {label}  (n = {len(sh)})")
    print("    t      shape<=t  depth<=t  both<=t")
    for t in THRESHOLDS:
        print(f"   {t:4.2f}     {100*(sh<=t).mean():6.1f}    {100*(dp<=t).mean():6.1f}   "
              f"{100*((sh<=t)&(dp<=t)).mean():6.1f}")
    q = lambda x: " / ".join(f"{v:.3f}" for v in np.percentile(x, [50, 75, 90]))
    print(f"    分位点(50/75/90)  shape {q(sh)}   depth {q(dp)}")


def distribution(D):
    print(f"\n[B] しきい値を振った分布  ({D.path})")
    b, m, kt, kp = D.cols(D.hit)
    sh, dp = D.shape_err(b, m, kp, kt), D.depth_err(b, m)
    sweep(sh, dp, "全体（ソース番号まで一致した列）")
    for name, cols in (("well-conditioned", CORE8), ("degenerate", CLUSTER), ("weak", WEAK)):
        sel = np.isin(m, cols)
        if sel.sum() >= 20:
            sweep(sh[sel], dp[sel], f"group = {name}")


# ---------------------------------------------------------------- C. 積の誤差
def product_error(D):
    print(f"\n[C] 見かけの変調 d*e(n) の誤差  ({D.path})")
    for label, mask in (("ソース番号まで一致した列", D.hit),
                        ("検出が当たった列（番号は問わない）", D.pred_on & D.true_on)):
        b, m, kt, kp = D.cols(mask)
        if len(b) == 0:
            continue
        a = D.d_pred[b, m][:, None] * D.e_param[b, kp]      # 予測の見かけ軌跡
        t = D.d_true[b, m][:, None] * D.e_true[b, kt]       # 真の見かけ軌跡
        mae = np.abs(a - t).mean(-1)
        denom = np.maximum((a * a).sum(-1), 1e-12)
        alpha = (a * t).sum(-1) / denom                     # 最適スカラー倍
        mae_s = np.abs(alpha[:, None] * a - t).mean(-1)
        q = lambda x: " / ".join(f"{v:.3f}" for v in np.percentile(x, [50, 75, 90]))
        print(f"\n  {label}  (n = {len(b)})")
        print(f"    そのまま        MAE 中央値/75/90 = {q(mae)}   <=0.1 の割合 {100*(mae<=TOL).mean():.1f}%")
        print(f"    最適倍率で補正  MAE 中央値/75/90 = {q(mae_s)}   <=0.1 の割合 {100*(mae_s<=TOL).mean():.1f}%")
        print(f"    最適倍率 alpha  中央値/75/90 = {q(alpha)}"
              f"   （1 より大きい = 予測の振れが小さい）")
        print(f"    補正で改善した割合 {100*(mae_s < mae - 1e-6).mean():.1f}%  "
              f"改善量の中央値 {np.median(mae - mae_s):.4f}")


# ---------------------------------------------------------------- D. 順序
def ordering(D):
    print(f"\n[D] ソース番号どおり vs 最適対応（ハンガリアン法）  ({D.path})")
    for name, sig in (("Stage 1 の軌跡 traj", D.traj), ("パラメータ経由 e_param", D.e_param)):
        if not np.any(sig):
            print(f"\n  {name}: 全て 0 のためスキップ")
            continue
        ident, opt, same_perm, n = [], [], 0, 0
        for b in range(D.B):
            u = np.nonzero(D.used[b])[0]
            if len(u) == 0:
                continue
            cost = np.abs(sig[b][:, None, :] - D.e_true[b][None, :, :]).mean(-1)  # [k_pred,k_true]
            r, c = linear_sum_assignment(cost[:, u])
            ident.extend(cost[u, u])
            opt.extend(cost[r, u[c]])
            same_perm += int(np.array_equal(u[c], r))
            n += 1
        ident, opt = np.array(ident), np.array(opt)
        print(f"\n  {name}  (使用中ソース {len(ident)} 本 / {n} サンプル)")
        print(f"    番号どおり  MAE 中央値 {np.median(ident):.4f}   <=0.1 の割合 {100*(ident<=TOL).mean():.2f}%")
        print(f"    最適対応    MAE 中央値 {np.median(opt):.4f}   <=0.1 の割合 {100*(opt<=TOL).mean():.2f}%")
        print(f"    最適対応が恒等だったサンプル {100*same_perm/max(n,1):.1f}%")


# ---------------------------------------------------------------- E. 深さの縮み
def shrinkage(D):
    b, m, kt, kp = D.cols(D.hit)
    p, t = D.d_pred[b, m], D.d_true[b, m]
    keep = np.abs(t) > 1e-6
    ratio = np.abs(p[keep]) / np.abs(t[keep])
    print(f"\n[E] 深さの縮み  ({D.path}, n = {len(p)})")
    print(f"    予測 平均 {p.mean():+.3f} 標準偏差 {p.std():.3f} / "
          f"真 平均 {t.mean():+.3f} 標準偏差 {t.std():.3f}")
    print(f"    標準偏差の比 {p.std()/max(t.std(),1e-9):.3f}   相関 {np.corrcoef(p,t)[0,1]:.3f}")
    print(f"    |予測|/|真| 中央値 {np.median(ratio):.3f}  "
          f"(25% {np.percentile(ratio,25):.3f} / 75% {np.percentile(ratio,75):.3f})")
    print(f"    符号一致率 {100*(np.sign(p[keep])==np.sign(t[keep])).mean():.1f}%")


# ---------------------------------------------------------------- F. 共通集合
def common(A, B_):
    key = lambda D: {(int(i), int(mm)) for i, mm in
                     zip(D.idx[np.nonzero(D.hit)[0]], np.nonzero(D.hit)[1])}
    ka, kb = key(A), key(B_)
    inter = ka & kb
    print(f"\n[F] 共通ヒット集合  {A.path}: {len(ka)} / {B_.path}: {len(kb)} / 共通 {len(inter)}")
    if not inter:
        return
    for D in (A, B_):
        b, m = np.nonzero(D.hit)
        sel = np.array([(int(D.idx[bb]), int(mm)) in inter for bb, mm in zip(b, m)])
        b, m = b[sel], m[sel]
        kt, kp = D.z_true[b, m], D.z_pred[b, m]
        sh, dp = D.shape_err(b, m, kp, kt), D.depth_err(b, m)
        print(f"\n  {D.path}  (n = {len(b)})")
        print(f"    shape_ok {100*(sh<=TOL).mean():.1f}%   depth_ok {100*(dp<=TOL).mean():.1f}%")
        q = lambda x: " / ".join(f"{v:.3f}" for v in np.percentile(x, [50, 75, 90]))
        print(f"    分位点(50/75/90)  shape {q(sh)}   depth {q(dp)}")


def gate_margin(D):
    """ゲートが ∅ とどれだけ拮抗しているか。拮抗が多いほど実行ごとに n_pred が揺れる。"""
    if D.margin is None:
        return
    mg = D.margin.ravel()
    print(f"\n[G] ゲートの余裕（1位のソース − ∅ のロジット差）  ({D.path})")
    print(f"    割り当て側（margin > 0）の列 {100*(mg>0).mean():.1f}%  "
          f"→ n_pred {D.pred_on.sum(1).mean():.3f}")
    for eps in (0.05, 0.1, 0.25, 0.5, 1.0):
        print(f"    |margin| < {eps:<4} の拮抗列 {100*(np.abs(mg)<eps).mean():5.2f}%  "
          f"（全 {len(mg)} 列中 {int((np.abs(mg)<eps).sum())} 列）")
    q = " / ".join(f"{v:+.2f}" for v in np.percentile(mg, [10, 50, 90]))
    print(f"    margin の分位点(10/50/90) {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    ds = [Dump(p) for p in args.dumps]
    for D in ds:
        basic_metrics(D); distribution(D); product_error(D); ordering(D)
        shrinkage(D); gate_margin(D)
    if len(ds) == 2:
        common(ds[0], ds[1])


if __name__ == "__main__":
    main()
