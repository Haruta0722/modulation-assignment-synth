r"""lfo_whatif.py — LFO の列で desc_rec が伸びない原因（実験 46 の補足）

  python lfo_whatif.py dump_v30_single_s{0,1,2}_test.npz

検出が当たった LFO の列（真も予測も LFO の枠）について、予測の (レート, 位相, morph) と深さを 1 つずつ正解に置き換えて
LFO の波形を作り直し、形状 OK（波形の平均絶対誤差 ≤ 0.1）・深さ OK・desc_rec への寄与がどれだけ上がるかを見る。
波形は diff_synth_tf.lfo と同じ式（lfo_phase_check.lfo_wave）をフレームの時刻で計算する。
"""
import argparse

import numpy as np

import analyze_dump as A
from lfo_phase_check import lfo_wave

SR, HOP, OFF, N, TOL = 16000, 256, 512, 24000, 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    variants = [("予測のまま", ()), ("位相を正解に", ("phase",)), ("レートを正解に", ("rate",)),
                ("morph を正解に", ("morph",)), ("位相とレートを正解に", ("phase", "rate")),
                ("位相・レート・morph を正解に", ("phase", "rate", "morph"))]
    res = {v[0]: [] for v in variants}
    extra = []
    for p in args.dumps:
        D = A.Dump(p)
        d = np.load(p)
        sp_t, sp_p = d["sp_true"], d["sp_pred"]
        T = D.e_true.shape[-1]
        t = np.minimum(np.arange(T) * HOP + OFF, N - 1) / SR
        b, m = np.nonzero(D.pred_on & D.true_on)
        kt, kp = D.z_true[b, m], D.z_pred[b, m]
        lfo = (D.type_true[b, kt] == 0) & (D.type_pred[b, kp] == 0)
        b, m, kt, kp = b[lfo], m[lfo], kt[lfo], kp[lfo]
        n_true = D.true_on.sum()
        dep_ok = np.abs(D.d_pred[b, m] - D.d_true[b, m]) <= TOL
        sign_ok = np.sign(D.d_pred[b, m]) == np.sign(D.d_true[b, m])
        e_true = D.e_true[b, kt]
        chk = np.abs(lfo_wave(sp_p[b, kp, 0], sp_p[b, kp, 1], sp_p[b, kp, 2], t) - D.e_param[b, kp]).max()
        for name, repl in variants:
            q = {j: sp_p[b, kp, i].copy() for i, j in enumerate(("rate", "phase", "morph"))}
            for j in repl:
                q[j] = sp_t[b, kt, ("rate", "phase", "morph").index(j)]
            shp_ok = np.abs(lfo_wave(q["rate"], q["phase"], q["morph"], t) - e_true).mean(1) <= TOL
            res[name].append([shp_ok.mean(), (dep_ok & shp_ok).sum() / n_true, shp_ok.sum() / n_true])
        extra.append([len(b) / n_true, dep_ok.mean(), sign_ok.mean(), dep_ok[sign_ok].mean(), chk])
    ex = np.array(extra).mean(0)
    print(f"検出が当たった LFO の列 / 真の配線 {ex[0]:.3f}。深さ OK {ex[1]:.3f}、深さの符号の正解率 {ex[2]:.3f}、"
          f"符号が合った列の深さ OK {ex[3]:.3f}（式で作り直した波形と e_param の差の最大 {ex[4]:.1e}）")
    print(f"\n{'':28s}{'形状 OK':>10s}{'desc_rec への寄与':>18s}{'（深さも正解なら）':>18s}")
    for name, _ in variants:
        v = np.array(res[name]).mean(0)
        print(f"  {name:26s}{v[0]:10.3f}{v[1]:18.4f}{v[2]:18.4f}")


if __name__ == "__main__":
    main()
