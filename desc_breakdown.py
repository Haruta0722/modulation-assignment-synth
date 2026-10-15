r"""desc_breakdown.py — desc_rec を ENV の列と LFO の列に分け、型・深さ・形状のどこで落ちるかを見る（実験 41）

  python desc_breakdown.py dump_v25_single_s{0,1,2}_test.npz

検出が当たった列（番号は問わない）を、真のソースの型で分けて:
  検出が当たった列 / 真の配線、型 OK、深さ OK（誤差 ≤ 0.1）、形状 OK（パラメータから作った e(n) の誤差 ≤ 0.1）、
  形状の誤差の中央値、d·e OK、desc_rec への寄与（型・深さ・形状すべて OK の列 / 真の配線）
"""
import argparse

import numpy as np

import analyze_dump as A

TOL = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    rows = {"ENV": [], "LFO": []}
    for p in args.dumps:
        D = A.Dump(p)
        b, m = np.nonzero(D.pred_on & D.true_on)
        kt, kp = D.z_true[b, m], D.z_pred[b, m]
        typ_ok = D.type_true[b, kt] == D.type_pred[b, kp]
        dep_ok = np.abs(D.d_pred[b, m] - D.d_true[b, m]) <= TOL
        shp_err = np.abs(D.e_param[b, kp] - D.e_true[b, kt]).mean(-1)
        shp_ok = shp_err <= TOL
        de = np.abs(D.d_pred[b, m, None] * D.e_param[b, kp] - D.d_true[b, m, None] * D.e_true[b, kt]).mean(-1)
        n_true = D.true_on.sum()
        for name, sel in (("ENV", D.type_true[b, kt] == 1), ("LFO", D.type_true[b, kt] == 0)):
            rows[name].append([sel.sum() / n_true, typ_ok[sel].mean(), dep_ok[sel].mean(), shp_ok[sel].mean(),
                               np.median(shp_err[sel]), (de <= TOL)[sel].mean(),
                               (typ_ok & dep_ok & shp_ok)[sel].sum() / n_true])
    print(f"{'':5s}{'当たった列/真の配線':>14s}{'型 OK':>8s}{'深さ OK':>9s}{'形状 OK':>9s}{'形状誤差':>9s}"
          f"{'d·e OK':>9s}{'desc_rec への寄与':>16s}")
    for name in ("ENV", "LFO"):
        v = np.array(rows[name]).mean(0)
        print(f"{name:5s}{v[0]:14.3f}{v[1]:8.3f}{v[2]:9.3f}{v[3]:9.3f}{v[4]:9.3f}{v[5]:9.3f}{v[6]:16.4f}")


if __name__ == "__main__":
    main()
