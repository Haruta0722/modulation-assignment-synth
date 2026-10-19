r"""error_profile.py — 予測の誤差の分布と、2 乗誤差での定数予測との比較（実験 48 の補足）

  python error_profile.py dump_v35_single_s{0,1,2}_test.npz

「許容誤差に入ったかどうか」だけでは、正解の近くまで来ていて少し外れているのか、遠く外れているのかが分からない。
検出が当たった列（真も予測も同じ型）について、
  2 乗誤差（MSE）と、定数予測（正解の平均。位相は円周上の平均）の 2 乗誤差、その比から作る「2 乗誤差の改善」
  （1 − MSE / 定数予測の MSE。1 で完全、0 で定数予測と同じ、負なら定数予測より悪い）
  誤差の大きさの分布（許容誤差の何倍の範囲に入っているか）
を、モデルと定数予測で並べる。
  LFO: レート（相対誤差。2 乗誤差は対数で）、位相（周期の割合、円周上）、深さ、形状（波形の平均絶対誤差）
  ENV: 深さ、形状
形状の許容誤差 0.1 は、振幅 0.5 の正弦波では位相のずれ約 0.05 周期に当たる（平均絶対誤差 = (2/π)|sin(πΔφ)|）。
レートの誤差は時間とともに位相のずれとして積み上がる（1.5 秒の中ほどで Δr × 0.75 秒）ので、
形状の許容誤差に収めるには、5 Hz で約 1〜2% のレートの精度が要る
"""
import argparse

import numpy as np

import analyze_dump as A

TOL = 0.1


def wrap(x):
    return (x + 0.5) % 1.0 - 0.5


def bins_of(err, edges):
    """err を edges で区切った範囲ごとの割合"""
    idx = np.searchsorted(edges, err, side="left")
    return np.bincount(idx, minlength=len(edges) + 1)[:len(edges) + 1] / max(len(err), 1)


ITEMS = {
    # 名前: (区切り, 区切りの表示)
    "LFO レート（相対誤差）": ([0.02, 0.05, 0.1, 0.3], ["≤2%", "2〜5%", "5〜10%", "10〜30%", ">30%"]),
    "LFO 位相（周期）": ([0.05, 0.1, 0.25], ["≤0.05", "0.05〜0.1", "0.1〜0.25", ">0.25"]),
    "LFO 深さ": ([0.1, 0.2, 0.4], ["≤0.1", "0.1〜0.2", "0.2〜0.4", ">0.4"]),
    "LFO 形状": ([0.1, 0.2, 0.3], ["≤0.1", "0.1〜0.2", "0.2〜0.3", ">0.3"]),
    "ENV 深さ": ([0.1, 0.2, 0.4], ["≤0.1", "0.1〜0.2", "0.2〜0.4", ">0.4"]),
    "ENV 形状": ([0.1, 0.2, 0.3], ["≤0.1", "0.1〜0.2", "0.2〜0.3", ">0.3"]),
}


def analyze(path):
    D = A.Dump(path)
    d = np.load(path)
    sp_t, sp_p = d["sp_true"], d["sp_pred"]
    b, m = np.nonzero(D.pred_on & D.true_on)
    kt, kp = D.z_true[b, m], D.z_pred[b, m]
    out = {}
    for typ, name in ((0, "LFO"), (1, "ENV")):
        s = (D.type_true[b, kt] == typ) & (D.type_pred[b, kp] == typ)
        bb, mm, tt, pp = b[s], m[s], kt[s], kp[s]
        # 深さ: 定数予測 = 正解の平均
        dt, dp = D.d_true[bb, mm], D.d_pred[bb, mm]
        out[f"{name} 深さ"] = (np.abs(dp - dt), np.abs(dt.mean() - dt), (dp - dt) ** 2, (dt.mean() - dt) ** 2)
        # 形状: 定数予測 = 正解の波形全体の平均値の水平線
        et, ep = D.e_true[bb, tt], D.e_param[bb, pp]
        c = et.mean()
        out[f"{name} 形状"] = (np.abs(ep - et).mean(1), np.abs(c - et).mean(1),
                               ((ep - et) ** 2).mean(1), ((c - et) ** 2).mean(1))
        if typ == 0:
            out["LFO 深さの符号の誤り"] = float(np.mean(np.sign(dp) != np.sign(dt)))
            rt, rp = sp_t[bb, tt, 0], sp_p[bb, pp, 0]
            lc = np.log(rt).mean()                                       # 定数予測: 対数の平均
            out["LFO レート（相対誤差）"] = (np.abs(rp - rt) / rt, np.abs(np.exp(lc) - rt) / rt,
                                         np.log(rp / rt) ** 2, (lc - np.log(rt)) ** 2)
            pt, pq = sp_t[bb, tt, 1], sp_p[bb, pp, 1]
            cm = np.angle(np.exp(2j * np.pi * pt).mean()) / (2 * np.pi)  # 定数予測: 円周上の平均
            out["LFO 位相（周期）"] = (np.abs(wrap(pq - pt)), np.abs(wrap(cm - pt)),
                                    wrap(pq - pt) ** 2, wrap(cm - pt) ** 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    R = [analyze(p) for p in args.dumps]
    print(f"LFO の列で深さの符号を誤った割合: {np.mean([r['LFO 深さの符号の誤り'] for r in R]):.3f}")
    for key, (edges, labels) in ITEMS.items():
        err = np.concatenate([r[key][0] for r in R])
        err_c = np.concatenate([r[key][1] for r in R])
        skill = [1 - r[key][2].mean() / r[key][3].mean() for r in R]
        print(f"\n[{key}]  {len(err)} 列（3 本の合計）  2 乗誤差の改善 {np.mean(skill):.3f} ± {np.std(skill, ddof=1):.3f}"
              f"  （MSE {np.mean([r[key][2].mean() for r in R]):.4f}、定数予測の MSE {np.mean([r[key][3].mean() for r in R]):.4f}）")
        print(f"  {'':10s}" + "".join(f"{l:>11s}" for l in labels) + f"{'中央値':>9s}")
        for name, e in (("モデル", err), ("定数予測", err_c)):
            print(f"  {name:10s}" + "".join(f"{v:11.3f}" for v in bins_of(e, edges)) + f"{np.median(e):9.3f}")


if __name__ == "__main__":
    main()
