r"""phase_from_traj.py — 単段の精密なレートと二段の軌跡を組み合わせて、LFO の位相を求める（実験 49）

  python phase_from_traj.py --single dump_v35_single_s{0,1,2}_test.npz --two dump_v36_two_s{0,1,2}_test.npz

単段（v35）はレートを精密に出せるが位相が読めない。二段の Stage 1 の軌跡は位相の手がかりを持つ（実験 36・41）。
そこで、単段が LFO の枠に出したレート（と morph）を固定し、位相だけを 0〜1 の 128 通りから選んで、
二段の同じ枠の軌跡と時間方向の相関が最大になる LFO の波形を探す（学習なしの型紙の当てはめ）。
単段の予測の位相だけをこれに置き換え、LFO の列の位相・形状・desc_rec がどう変わるかを見る。

使うレート（比較）
  単段のレート      : v35 の予測（本命）
  真のレート        : 正解（上限の目安）
  二段のレート      : 二段の Stage 2 の予測
位相をそのまま比べる参考: 単段の位相（置き換えない）、二段の Stage 2 の位相
評価は desc_breakdown と同じ「検出が当たった列」（単段の予測で、真も予測も同じ型）。desc_rec は ENV の列の寄与（変わらない）と
LFO の列の寄与の合計
"""
import argparse

import numpy as np

import analyze_dump as A
from lfo_phase_check import dump_tri, lfo_wave

SR, HOP, OFF, N, TOL, NPH = 16000, 256, 512, 24000, 0.1, 128


def wrap(x):
    return (x + 0.5) % 1.0 - 0.5


def fit_phase(traj, rate, morph, t, tri, chunk=1000):
    """traj [n,T] と相関が最大になる位相（rate・morph は [n] で固定）"""
    phases = np.arange(NPH) / NPH
    out = np.zeros(len(traj))
    for i in range(0, len(traj), chunk):
        w = traj[i:i + chunk]
        n = len(w)
        wc = w - w.mean(1, keepdims=True)
        wc /= np.maximum(np.linalg.norm(wc, axis=1, keepdims=True), 1e-9)
        bank = lfo_wave(np.repeat(rate[i:i + n], NPH), np.tile(phases, n), np.repeat(morph[i:i + n], NPH),
                        t, tri).reshape(n, NPH, -1)
        bc = bank - bank.mean(2, keepdims=True)
        bc /= np.maximum(np.linalg.norm(bc, axis=2, keepdims=True), 1e-9)
        out[i:i + n] = phases[np.argmax(np.einsum("nt,npt->np", wc, bc), 1)]
    return out


def analyze(p_single, p_two):
    S, Tw = A.Dump(p_single), A.Dump(p_two)
    ds_, dt_ = np.load(p_single), np.load(p_two)
    assert np.array_equal(S.idx, Tw.idx), "単段と二段の dump の音の並びが違う"
    tri = dump_tri(ds_)
    assert tri == dump_tri(dt_)
    T = S.e_true.shape[-1]
    t = np.minimum(np.arange(T) * HOP + OFF, N - 1) / SR
    sp_t, sp_s, sp_2 = ds_["sp_true"], ds_["sp_pred"], dt_["sp_pred"]
    b, m = np.nonzero(S.pred_on & S.true_on)
    kt, kp = S.z_true[b, m], S.z_pred[b, m]
    n_true = S.true_on.sum()
    typ_ok = S.type_true[b, kt] == S.type_pred[b, kp]
    dep_ok = np.abs(S.d_pred[b, m] - S.d_true[b, m]) <= TOL
    env = (S.type_true[b, kt] == 1) & typ_ok
    shp_env = np.abs(S.e_param[b[env], kp[env]] - S.e_true[b[env], kt[env]]).mean(1) <= TOL
    desc_env = (dep_ok[env] & shp_env).sum() / n_true
    lfo = (S.type_true[b, kt] == 0) & typ_ok
    bl, tl, pl = b[lfo], kt[lfo], kp[lfo]
    e_true = S.e_true[bl, tl]
    ph_true = sp_t[bl, tl, 1]
    traj = Tw.traj[bl, pl]
    r_s, m_s, r_2, r_t = sp_s[bl, pl, 0], sp_s[bl, pl, 2], sp_2[bl, pl, 0], sp_t[bl, tl, 0]
    # (波形を作るレート, 位相)。当てはめに使ったレートで波形も作る
    variants = {
        "単段のまま（置き換えない）": (r_s, sp_s[bl, pl, 1]),
        "二段の Stage 2 の位相（レートは単段）": (r_s, sp_2[bl, pl, 1]),
        "軌跡に当てはめ（二段のレート）": (r_2, fit_phase(traj, r_2, m_s, t, tri)),
        "軌跡に当てはめ（単段のレート）": (r_s, fit_phase(traj, r_s, m_s, t, tri)),
        "位相を正解に（レートは単段、参考）": (r_s, ph_true),
        "軌跡に当てはめ（真のレート、上限の目安）": (r_t, fit_phase(traj, r_t, m_s, t, tri)),
    }
    out = {"n_lfo": len(bl), "desc_env": desc_env, "rate": sp_t[bl, tl, 0]}
    for name, (r, ph) in variants.items():
        e = lfo_wave(r, ph, m_s, t, tri)
        shp = np.abs(e - e_true).mean(1)
        dph = np.abs(wrap(ph - ph_true))
        out[name] = dict(agree=np.cos(2 * np.pi * (ph - ph_true)), dph=dph, shp_ok=shp <= TOL, shp=shp,
                         desc_lfo=(dep_ok[lfo] & (shp <= TOL)).sum() / n_true)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", nargs="+", required=True)
    ap.add_argument("--two", nargs="+", required=True)
    args = ap.parse_args()
    R = [analyze(a, b) for a, b in zip(args.single, args.two)]
    names = [k for k in R[0] if isinstance(R[0][k], dict)]
    print(f"検出が当たった LFO の列: {sum(r['n_lfo'] for r in R)} 列（{len(R)} 本の合計）。"
          f"ENV の列の desc_rec への寄与 {np.mean([r['desc_env'] for r in R]):.4f}")
    print(f"\n{'':40s}{'位相の一致度':>10s}{'1〜3 Hz':>9s}{'3〜10 Hz':>9s}{'|Δφ|≤0.05':>11s}"
          f"{'形状 OK':>9s}{'形状誤差':>9s}{'desc_rec':>10s}")
    for name in names:
        ag, a13, a310, d5, sok, shp, desc = [], [], [], [], [], [], []
        for r in R:
            v, rt = r[name], r["rate"]
            ag.append(v["agree"].mean())
            a13.append(v["agree"][(rt >= 1) & (rt < 3)].mean())
            a310.append(v["agree"][rt >= 3].mean())
            d5.append((v["dph"] <= 0.05).mean())
            sok.append(v["shp_ok"].mean())
            shp.append(np.median(v["shp"]))
            desc.append(r["desc_env"] + v["desc_lfo"])
        print(f"  {name:38s}{np.mean(ag):10.3f}{np.mean(a13):9.3f}{np.mean(a310):9.3f}{np.mean(d5):11.3f}"
              f"{np.mean(sok):9.3f}{np.mean(shp):9.3f}{np.mean(desc):10.4f} ± {np.std(desc, ddof=1):.4f}")


if __name__ == "__main__":
    main()
