r"""consistent_metrics.py — 1 サンプルにつき 1 つのソース対応 π で全指標を測る

  python consistent_metrics.py dump_v11_single_s{0,1,2}_test.npz

ソース番号は正準順序という約束事のラベルにすぎないため、各サンプルで
「予測ソース k → 真のソース π(k)」を 1 つ決め、配線・深さ・形状・desc_rec をすべてその π で数える。
π の決め方を 3 通り比べる。
  番号どおり : π = 恒等（従来の F1 と同じ。順序推定まで含めた難易度）
  形状で対応 : 使用中の真のソースについて、ソース信号 e(n) の MAE の和が最小になる π（本命）
  配線最大   : 一致する配線の数が最大になる π（採点対象を見て選ぶので楽観的な上限）
"""
import argparse
from itertools import permutations

import numpy as np

import analyze_dump as A

K, TOL = A.K, A.TOL
PERMS = np.array(list(permutations(range(K))))       # [24,K]、先頭が恒等（同点なら恒等を選ぶ）
INV = np.argsort(PERMS, axis=1)                        # 真 t → 予測 k


def choose_pi(D, how):
    B = D.B
    if how == "ident":
        return np.zeros(B, int)
    if how == "shape":
        cost = np.abs(D.e_param[:, :, None, :] - D.e_true[:, None, :, :]).mean(-1)   # [B,kp,kt]
        # total[b,p] = Σ_t used[b,t] · cost[b, INV[p,t], t]
        c = cost[:, INV, np.arange(K)[None, :]]                                       # [B,24,K]
        return np.argmin((c * D.used[:, None, :]).sum(-1), axis=1)
    if how == "wiring":
        zp = np.minimum(D.z_pred, K - 1)
        mapped = PERMS[:, zp]                                                         # [24,B,M]
        match = (mapped == D.z_true[None]) & D.pred_on[None]
        return np.argmax(match.sum(-1), axis=0)
    raise ValueError(how)


def metrics(D, how):
    pi = PERMS[choose_pi(D, how)]                                   # [B,K] 予測 k → 真
    zmap = np.where(D.pred_on, np.take_along_axis(pi, np.minimum(D.z_pred, K - 1), 1), -9)
    hit = D.true_on & (zmap == D.z_true)
    n_pred, n_true, tp = D.pred_on.sum(), D.true_on.sum(), hit.sum()
    P, R = tp / n_pred, tp / n_true
    b, m = np.nonzero(hit)
    kp, kt = D.z_pred[b, m], D.z_true[b, m]
    sh = np.abs(D.e_param[b, kp] - D.e_true[b, kt]).mean(-1)
    dp = np.abs(D.d_pred[b, m] - D.d_true[b, m])
    ty = D.type_pred[b, kp] == D.type_true[b, kt]
    de = np.abs(D.d_pred[b, m][:, None] * D.e_param[b, kp]
                - D.d_true[b, m][:, None] * D.e_true[b, kt]).mean(-1)
    ok = (sh <= TOL) & (dp <= TOL) & ty
    return dict(P=P, R=R, F1=2 * P * R / (P + R), depth_ok=(dp <= TOL).mean(),
                shape_ok=(sh <= TOL).mean(), type_ok=ty.mean(), de_ok=(de <= TOL).mean(),
                desc_rec=ok.sum() / n_true, ident=(np.all(pi == np.arange(K), 1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="同じ条件の seed 違いの dump（平均±標準偏差を出す）")
    args = ap.parse_args()
    Ds = [A.Dump(p) for p in args.dumps]
    keys = ["P", "R", "F1", "depth_ok", "shape_ok", "type_ok", "de_ok", "desc_rec", "ident"]
    print(f"{len(Ds)} dumps: {', '.join(args.dumps)}")
    print(f"{'':10s}" + "".join(f"{k:>16s}" for k in keys))
    for how, label in (("ident", "番号どおり"), ("shape", "形状で対応"), ("wiring", "配線最大")):
        rs = np.array([[metrics(D, how)[k] for k in keys] for D in Ds])
        sd = rs.std(0, ddof=1) if len(Ds) > 1 else np.zeros(len(keys))
        print(f"{label:10s}" + "".join(f"{mu:9.4f}±{s:.4f}" for mu, s in zip(rs.mean(0), sd)))
    # 検出（ソースを問わない）は π に依存しない
    det = np.array([A.f1((D.pred_on & D.true_on).sum(), (D.pred_on & ~D.true_on).sum(),
                         (~D.pred_on & D.true_on).sum()) for D in Ds])
    print(f"F1_detect {det.mean():.4f}±{det.std(ddof=1) if len(Ds) > 1 else 0:.4f}")


if __name__ == "__main__":
    main()
