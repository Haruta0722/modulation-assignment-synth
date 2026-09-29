r"""posthoc_calib.py — 再学習なしの後処理で、深さの縮みとゲートの付け足りなさを検証する

  python posthoc_calib.py dump_v10_single_s0_val.npz dump_v10_single_s0_test.npz

val で補正量を 1 つだけ決め、test にそのまま当てて測る（test は選択に使わない）。
  1. 深さの一律倍率 α:   d' = clip(α·d̂, -1, 1)
  2. ゲートのしきい値 τ:  margin > τ なら ∅ を除いた 1 位のソースを割り当てる（τ=0 が現状）
"""
import argparse

import numpy as np

import analyze_dump as A

K, TOL = A.K, A.TOL
ALPHAS = np.round(np.arange(1.0, 4.01, 0.1), 2)
TAUS = np.round(np.arange(-3.0, 1.01, 0.1), 2)


def apply(D, alpha=1.0, tau=0.0):
    """補正後の予測に差し替えた Dump を返す（元は変更しない）"""
    E = object.__new__(A.Dump)
    E.__dict__.update(D.__dict__)
    on = D.margin > tau
    E.z_pred = np.where(on, D.z_src, K)
    E.d_pred = np.where(on, np.clip(alpha * D.depth_raw, -1, 1), 0.0)
    E.pred_on = E.z_pred < K
    E.hit = E.true_on & E.pred_on & (E.z_true == E.z_pred)
    return E


def metrics(D):
    tp = D.hit.sum(); fp = (D.pred_on & ~D.hit).sum(); fn = (D.true_on & ~D.hit).sum()
    both = D.pred_on & D.true_on
    b, m, kt, kp = D.cols(D.hit)
    sh, dp = D.shape_err(b, m, kp, kt), D.depth_err(b, m)
    de = np.abs(D.d_pred[b, m][:, None] * D.e_param[b, kp]
                - D.d_true[b, m][:, None] * D.e_true[b, kt]).mean(-1)
    b2, m2, kt2, kp2 = D.cols(both)
    good = ((D.shape_err(b2, m2, kp2, kt2) <= TOL) & (D.depth_err(b2, m2) <= TOL)
            & (D.type_pred[b2, kp2] == D.type_true[b2, kt2]))
    return dict(F1=A.f1(tp, fp, fn), P=tp / max(tp + fp, 1), R=tp / max(tp + fn, 1),
                F1_detect=A.f1(both.sum(), (D.pred_on & ~D.true_on).sum(),
                               (~D.pred_on & D.true_on).sum()),
                n_pred=D.pred_on.sum(1).mean(),
                depth_ok=(dp <= TOL).mean(), shape_ok=(sh <= TOL).mean(),
                de_ok=(de <= TOL).mean(),
                desc_rec=good.sum() / max(D.true_on.sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("val")
    ap.add_argument("test")
    args = ap.parse_args()
    V, T = A.Dump(args.val), A.Dump(args.test)
    for D in (V, T):
        d = np.load(D.path)
        D.z_src, D.depth_raw = d["z_src"], d["depth_raw"]

    # 深さ: depth_ok 基準と d·e 基準の 2 通りで α を決める（ゲートは現状のまま）
    alpha_d = max(ALPHAS, key=lambda a: metrics(apply(V, alpha=a))["depth_ok"])
    alpha_de = max(ALPHAS, key=lambda a: metrics(apply(V, alpha=a))["de_ok"])
    # ゲート: F1 を最大にする τ（深さは現状のまま）
    tau = max(TAUS, key=lambda t: metrics(apply(V, tau=t))["F1"])

    rows = [("現状", apply(T)),
            (f"深さ α={alpha_d}(depth基準)", apply(T, alpha=alpha_d)),
            (f"深さ α={alpha_de}(d·e基準)", apply(T, alpha=alpha_de)),
            (f"ゲート τ={tau}", apply(T, tau=tau))]
    keys = ["F1", "P", "R", "F1_detect", "n_pred", "depth_ok", "shape_ok", "de_ok", "desc_rec"]
    print(f"test: {args.test}（α・τ は val {args.val} で決定）")
    print(f"{'':26s}" + "".join(f"{k:>10s}" for k in keys))
    for name, D in rows:
        r = metrics(D)
        print(f"{name:26s}" + "".join(f"{r[k]:10.4f}" for k in keys))

    print("\n[val] ゲートしきい値 τ を振ったとき")
    print(f"{'τ':>6s}" + "".join(f"{k:>10s}" for k in ("F1", "P", "R", "n_pred", "F1_detect")))
    for t in (-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5):
        r = metrics(apply(V, tau=t))
        print(f"{t:6.2f}" + "".join(f"{r[k]:10.4f}" for k in ("F1", "P", "R", "n_pred", "F1_detect")))
    print("\n[val] 深さの倍率 α を振ったとき（ゲートは現状）")
    print(f"{'α':>6s}" + "".join(f"{k:>10s}" for k in ("depth_ok", "de_ok", "desc_rec")))
    for a in (1.0, 1.5, 2.0, 2.5, 3.0):
        r = metrics(apply(V, alpha=a))
        print(f"{a:6.2f}" + "".join(f"{r[k]:10.4f}" for k in ("depth_ok", "de_ok", "desc_rec")))


if __name__ == "__main__":
    main()
