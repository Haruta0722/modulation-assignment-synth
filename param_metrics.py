r"""param_metrics.py — パラメータごとに「音から推定できているか」を定数予測と比べて測る

  python param_metrics.py dump_v13_single_s{0,1,2}_test.npz

各パラメータについて、真と予測の組を集めて次を出す（複数の dump を渡すと平均 ± 標準偏差）。
  相関       : 予測と真のピアソン相関。音から情報を読み取れているか（0 なら読み取れていない）
  相関(同型1つ): 同じ型の使用中ソースが 1 つだけの音での相関。正準順序では枠の番号自体が
               アタック / レートの順位を表すため、全体の相関はその分だけ高く出る。これを除いた値
  ばらつき比 : 予測の標準偏差 / 真の標準偏差。平均側に縮んでいないか（1 なら縮みなし）
  合格率     : |予測 − 真| ≤ しきい値 の割合。しきい値 = frac × 真の標準偏差（既定 0.25）
  定数の合格率: 常に真の中央値を答えた場合の合格率（入力を使わない場合の基準）
  誤差の改善 : 1 − MAE / 定数予測の MAE（0 なら定数と同じ、1 なら完全）
LFO のレートは対数で、位相は円周上の差で測る（位相の相関・ばらつき比は出さない）。

組の作り方
  深さ     : ソース番号まで当たった列（ゲート前の深さの出力と、列の真の深さ）
  ENV / LFO: 使用中の真のソース t と、対応する予測の枠（--match ident なら同じ番号）の型が一致したもの
  θ_stat   : 全サンプル × 全変調先
"""
import argparse

import numpy as np

import analyze_dump as A
import order_analysis as O

K = A.K
ENV_NAMES = ["attack", "hold", "decay", "sustain", "release"]


def env_frac_np(p, t_off=1.0):
    """estimator.env_frac と同じ割合空間（dataset_v7 は t_off = 1.0）"""
    a, h, d = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([a / t_off, h / np.maximum(t_off - a, 1e-3),
                     d / np.maximum(t_off - a - h, 1e-3)], 1).clip(0, 2)


def wrap(x):
    return (x + 0.5) % 1.0 - 0.5


def stats(t, p, kind, frac):
    if kind == "log":
        t, p = np.log(np.maximum(t, 1e-6)), np.log(np.maximum(p, 1e-6))
    if kind == "circ":
        c = np.angle(np.exp(2j * np.pi * t).mean()) / (2 * np.pi)
        err, err_c = np.abs(wrap(p - t)), np.abs(wrap(c - t))
        sd_t = np.sqrt((wrap(t - c) ** 2).mean())
        corr = ratio = np.nan
    else:
        err, err_c = np.abs(p - t), np.abs(np.median(t) - t)
        sd_t = t.std()
        corr = np.corrcoef(p, t)[0, 1] if p.std() > 0 else 0.0
        ratio = p.std() / sd_t
    thr = frac * sd_t
    return dict(n=len(t), corr=corr, ratio=ratio, ok=(err <= thr).mean(),
                ok_const=(err_c <= thr).mean(), skill=1 - err.mean() / err_c.mean())


def pairs(D, match):
    out = {}
    b, m = np.nonzero(D.hit)
    out["depth"] = ("lin", D.d_true[b, m], D.d_pred[b, m], None)
    pi = (np.tile(np.arange(D.K), (D.B, 1)) if match == "ident"
          else O.wiring_pi(D.z_pred, D.z_true, D.pred_on))
    inv = np.argsort(pi, axis=1)                                      # 真 t → 予測の枠
    bb, tt = np.nonzero(D.used)
    kk = inv[bb, tt]
    same = D.type_true[bb, tt] == D.type_pred[bb, kk]
    env = same & (D.type_true[bb, tt] == 1)
    lfo = same & (D.type_true[bb, tt] == 0)
    # 同じ型の使用中ソースが 1 つだけの音: 枠の番号が順位の手がかりにならない
    n_env = (D.used & (D.type_true == 1)).sum(1)
    n_lfo = (D.used & (D.type_true == 0)).sum(1)
    one = np.where(D.type_true[bb, tt] == 1, n_env[bb] == 1, n_lfo[bb] == 1)
    for j, name in enumerate(ENV_NAMES):
        out[f"ENV {name}"] = ("lin", D.sp_true[bb[env], tt[env], j], D.sp_pred[bb[env], kk[env], j],
                              one[env])
    # hold・decay は「残り時間に対する割合」でも測る（秒は前段の誤差を引き継ぐ）
    ft, fp = env_frac_np(D.sp_true[bb[env], tt[env]]), env_frac_np(D.sp_pred[bb[env], kk[env]])
    for j, name in ((1, "hold(割合)"), (2, "decay(割合)")):
        out[f"ENV {name}"] = ("lin", ft[:, j], fp[:, j], one[env])
    for j, (name, kind) in enumerate((("rate", "log"), ("phase", "circ"), ("morph", "lin"))):
        out[f"LFO {name}"] = (kind, D.sp_true[bb[lfo], tt[lfo], j], D.sp_pred[bb[lfo], kk[lfo], j],
                              one[lfo])
    out["θ_stat"] = ("lin", D.stat_true.ravel(), D.stat_pred.ravel(), None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--match", choices=["ident", "wiring"], default="ident",
                    help="真と予測のソースの対応（ident = 番号どおり、wiring = 配線の一致が最大）")
    ap.add_argument("--frac", type=float, default=0.25, help="しきい値 = frac × 真の標準偏差")
    args = ap.parse_args()

    res = {}
    for path in args.dumps:
        D = O.load(path)
        d = np.load(path)
        D.stat_true, D.stat_pred = d["stat_true"], d["stat_pred"]
        for name, (kind, t, p, one) in pairs(D, args.match).items():
            r = stats(t, p, kind, args.frac)
            r["corr1"] = stats(t[one], p[one], kind, args.frac)["corr"] if one is not None else np.nan
            res.setdefault(name, []).append(r)

    keys = [("corr", "相関"), ("corr1", "相関(同型1つ)"), ("ratio", "ばらつき比"), ("ok", "合格率"),
            ("ok_const", "定数の合格率"), ("skill", "誤差の改善")]
    print(f"{len(args.dumps)} dumps / 対応 {args.match} / しきい値 {args.frac} × 真の標準偏差")
    print(f"{'パラメータ':14s}{'組の数':>8s}" + "".join(f"{lab:>16s}" for _, lab in keys))
    for name, rs in res.items():
        n = int(np.mean([r["n"] for r in rs]))
        row = f"{name:14s}{n:8d}"
        for k, _ in keys:
            v = np.array([r[k] for r in rs], float)
            if np.all(np.isnan(v)):
                row += f"{'—':>16s}"
            else:
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                row += f"{v.mean():10.3f}±{sd:.3f}"
        print(row)


if __name__ == "__main__":
    main()
