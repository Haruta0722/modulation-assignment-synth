r"""perceptual_desc.py — 変調先ごとの許容誤差 τ_m（聞き分けの限界）で desc_rec を測り直す

  python perceptual_desc.py --calib calibrate_dest.npz dump_v18_single_s{0,1,2}_test.npz

τ_m: calibrate_dest.py の結果から、知覚距離（--dist、既定は JTFS 相対・PNP の設定）が聞き分けの限界 D* を
超える割合が 50% になる d·e(n) の誤差。試した範囲で 50% に届かなかった変調先は、試した最大の誤差を
τ_m とする（本当の τ_m はそれ以上なので控えめな値）。

知覚的な desc_rec: 検出が当たった列（ソース番号は問わない）で、見かけの変調 d̂·ê(n) と d·e(n) の
平均絶対差が τ_m 以下なら合格。合格数 ÷ 真の配線数。
比較: 同じ列で「変調なし」（d̂·ê = 0）と答えた場合の合格率。
"""
import argparse

import numpy as np

import analyze_dump as A
import diff_synth_tf as ds

K, M = ds.K_ENVS, ds.M_DEST
NAMES = [d[0] for d in ds.dest_specs(16000)]


def taus(cal, dist, kinds=("depth", "env")):
    Dstar = float(cal[f"Dstar_{dist}"][2])
    m_, kind, err, d = cal["m"], cal["kind"], cal["de_err"], cal[dist]
    out = []
    for m in range(M):
        sel = (m_ == m) & np.isin(kind, kinds)
        e, v = err[sel], d[sel]
        if len(e) < 10:
            out.append((np.nan, "?"))
            continue
        edges =np.unique(np.quantile(e, np.linspace(0, 1, 9)))
        xs, ps = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (e >= lo) & (e <= hi)
            if s.sum() >= 5:
                xs.append(np.median(e[s]))
                ps.append(np.mean(v[s] > Dstar))
        xs, ps = np.array(xs), np.maximum.accumulate(np.array(ps))
        if ps[0] >= 0.5:
            t, note = xs[0], "≤"                      # 最小の誤差ですでに半数が超える
        elif ps[-1] < 0.5:
            t, note = e.max(), "≥"                    # 試した範囲では届かない
        else:
            i = np.argmax(ps >= 0.5)
            t, note = float(np.interp(0.5, ps[i - 1:i + 1], xs[i - 1:i + 1])), ""
        out.append((float(t), note))
    return Dstar, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--calib", default="calibrate_dest.npz",
                    help="カンマ区切りで複数。後のファイルにある変調先は後の値で上書き（細かく測り直したもの）")
    ap.add_argument("--dist", default="jtfs_pnp_rel")
    args = ap.parse_args()

    tl = None
    for path in args.calib.split(","):
        Dstar, t2 = taus(np.load(path), args.dist)
        tl = t2 if tl is None else [b if not np.isnan(b[0]) else a for a, b in zip(tl, t2)]
    tau = np.array([t for t, _ in tl])
    print(f"距離 {args.dist}、D* = {Dstar:.4g}")
    print("τ_m: " + ", ".join(f"{NAMES[m]} {n}{t:.3f}" for m, (t, n) in enumerate(tl)))

    res = []                                          # dump ごと: dict
    per_dest = []
    for path in args.dumps:
        D = A.Dump(path)
        both = D.pred_on & D.true_on
        b, m = np.nonzero(both)
        kp, kt = D.z_pred[b, m], D.z_true[b, m]
        a = D.d_pred[b, m][:, None] * D.e_param[b, kp]
        t = D.d_true[b, m][:, None] * D.e_true[b, kt]
        err = np.abs(a - t).mean(-1)
        err0 = np.abs(t).mean(-1)                     # 変調なしと答えた場合
        n_true = D.true_on.sum()
        r = dict(n_true=n_true, recall=len(b) / n_true)
        for f in (1, 2):
            ok, ok0 = err <= f * tau[m], err0 <= f * tau[m]
            r[f"desc{f}"] = ok.sum() / n_true
            r[f"det{f}"] = ok.mean()
            r[f"zero{f}"] = ok0.mean()
        b2, m2, kt2, kp2 = D.cols(both)
        sh =D.shape_err(b2, m2, kp2, kt2) <= A.TOL
        dp = D.depth_err(b2, m2) <= A.TOL
        ty = D.type_pred[b2, kp2] == D.type_true[b2, kt2]
        r["old_desc"] = (sh & dp & ty).sum() / n_true
        res.append(r)
        per_dest.append([(np.mean(err[m == j] <= tau[j]) if (m == j).any() else np.nan,
                          np.mean(err0[m == j] <= tau[j]) if (m == j).any() else np.nan,
                          (m == j).sum()) for j in range(M)])

    mean = lambda k: np.mean([r[k] for r in res])
    sd = lambda k: np.std([r[k] for r in res], ddof=1) if len(res) > 1 else 0.0
    print(f"\n{len(args.dumps)} 本の平均 ± 標準偏差")
    print(f"  検出の再現率（真の配線のうち検出が当たった割合）  {mean('recall'):.3f} ± {sd('recall'):.3f}")
    print(f"  従来の desc_rec（深さ・形状 0.1、型）          {mean('old_desc'):.4f} ± {sd('old_desc'):.4f}")
    for f in (1, 2):
        print(f"  [許容誤差 {f}×τ_m]")
        print(f"    知覚的な desc_rec（合格 ÷ 真の配線）          {mean(f'desc{f}'):.4f} ± {sd(f'desc{f}'):.4f}")
        print(f"    検出が当たった列での合格率: モデル {mean(f'det{f}'):.3f} / 変調なしと答えた場合 {mean(f'zero{f}'):.3f}")

    pd = np.array(per_dest)                          # [dump, M, 3]
    print(f"\n変調先ごと（許容誤差 1×τ_m、検出が当たった列での合格率。{len(args.dumps)} 本の平均）")
    print(f"  {'変調先':10s}{'τ_m':>9s}{'件数':>7s}{'モデル':>9s}{'変調なし':>10s}")
    for j in range(M):
        t, n = tl[j]
        print(f"  {NAMES[j]:10s}{n + format(t, '.3f'):>9s}{int(pd[:, j, 2].mean()):7d}"
              f"{np.nanmean(pd[:, j, 0]):9.3f}{np.nanmean(pd[:, j, 1]):10.3f}")


if __name__ == "__main__":
    main()
