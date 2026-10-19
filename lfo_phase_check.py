r"""lfo_phase_check.py — LFO の位相が推定できない原因の切り分け

  python lfo_phase_check.py dump_v13_two_s{0,1,2}_test.npz            # 二段（Stage 1 の軌跡あり）
  python lfo_phase_check.py dump_v18_single_s{0,1,2}_test.npz         # 単段（パラメータのみ）

使用中の真の LFO ソース（番号どおりの枠と対応させる）について、
  [A] 軌跡（二段の Stage 1 がパラメータを通さず直接出す e(n)）と、パラメータから作った e(n) の再現度
      波形の相関（時間方向）、平均絶対誤差、定数予測（LFO の波形の平均値の水平線）との比較
  [B] 軌跡に最もよく合う位相: 真のレートと morph を固定し、位相だけを 0〜1 の 128 通りから探す
      → 軌跡そのものが位相の情報を持っているか
  [B2] 型紙の当てはめ: 真の値を一切使わず、レート・位相・morph の格子の LFO 波形のうち軌跡と相関が最大のものを選ぶ
      → 学習なしの単純な復号で、軌跡からどこまでパラメータに戻せるか（Stage 2 が超えるべき下限）
  [C] パラメータの位相・レート・morph の推定（予測の型が LFO の枠のみ）
  [D] 時刻 t での位相（φ + レート × t）の一致度: 位相が t=0 では外れても、レートと絡んで途中の時刻で合っているか
  [E] 変調先（そのソースの |真の深さ| が最大の列）と |深さ| ごとの位相の一致度（レート 1 Hz 以上）
      相関（位相は円周上の一致度 = cos(2π × 位相差) の平均）、ばらつき比、定数予測と比べた誤差の改善
位相は円周上の量なので、
  一致度     : mean cos(2π(φ̂ − φ))。1 で完全、0 で無関係
  ばらつき比 : 予測の円周上の分散 / 真の円周上の分散（1 − 平均ベクトルの長さ）
  誤差の改善 : 1 − 円周上の平均絶対誤差 / 定数予測の平均絶対誤差
"""
import argparse

import numpy as np

import analyze_dump as A
import diff_synth_tf as ds

SR, HOP, OFF = 16000, 256, 512
RATE_BINS = [(0.1, 1.0), (1.0, 3.0), (3.0, 10.01)]


def wrap(x):
    return (x + 0.5) % 1.0 - 0.5


def lfo_wave(rate, phase, morph, t, tri="v7"):
    """diff_synth_tf.lfo と同じ波形（rate, phase, morph は [n]、t は [T] 秒）→ [n,T]。
    tri: LFO の三角波の式（"v7" = dataset_v7 の誤った式、"tri" = 正しい三角波、実験 47）"""
    th = rate[:, None] * t[None, :] + phase[:, None]
    sin_w = np.sin(2 * np.pi * th)
    tri_w = 1.0 - 4.0 * np.abs(th + 0.25 - np.floor(th + {"v7": 0.75, "tri": 0.25}[tri]) - 0.5)
    m = morph[:, None]
    return 0.5 * (1.0 + m * sin_w + (1.0 - m) * tri_w)


def circ_stats(p, t):
    err = np.abs(wrap(p - t))
    c = np.angle(np.exp(2j * np.pi * t).mean()) / (2 * np.pi)          # 定数予測: 真の位相の円周平均
    err_c = np.abs(wrap(c - t))
    var = lambda a: 1 - np.abs(np.exp(2j * np.pi * a).mean())
    return dict(agree=float(np.mean(np.cos(2 * np.pi * (p - t)))),
                # 半周期ずれ（φ と φ+0.5）を同一視した一致度: LFO は φ+0.5 で上下反転するので、深さの符号と交換できる
                agree_half=float(np.mean(np.cos(4 * np.pi * (p - t)))),
                ratio=float(var(p) / max(var(t), 1e-9)),
                skill=float(1 - err.mean() / err_c.mean()))


def lin_stats(p, t):
    err, err_c = np.abs(p - t), np.abs(np.median(t) - t)
    return dict(agree=float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 else 0.0,
                ratio=float(p.std() / t.std()), skill=float(1 - err.mean() / err_c.mean()))


def wave_stats(w, e, const):
    """軌跡 w と真の波形 e（[n,T]）: 時間方向の相関の中央値、平均絶対誤差の中央値、定数予測と比べた改善"""
    wc, ec = w - w.mean(1, keepdims=True), e - e.mean(1, keepdims=True)
    corr = (wc * ec).sum(1) / np.maximum(np.sqrt((wc ** 2).sum(1) * (ec ** 2).sum(1)), 1e-12)
    mae = np.abs(w - e).mean(1)
    mae_c = np.abs(const - e).mean(1)
    return dict(corr=float(np.median(corr)), mae=float(np.median(mae)),
                skill=float(1 - mae.mean() / mae_c.mean()),
                amp=float(np.median(w.std(1) / np.maximum(e.std(1), 1e-9))),     # 振れ幅の比（1 で縮みなし）
                abscorr=float(np.median(np.abs(corr))))                        # 上下反転も一致とみなした相関


def dump_tri(d):
    """dump に記録された LFO の三角波の式（記録がない dump は dataset_v7 の式）"""
    return str(d["lfo_tri"]) if "lfo_tri" in d else "v7"


def template_fit(w, t, n_rate=160, n_phase=128, morphs=(0.0, 0.5, 1.0), tri="v7"):
    """軌跡 w [n,T] と時間方向の相関が最大になる LFO 波形の (レート, 位相, morph)。真の値は使わない"""
    wc = w - w.mean(1, keepdims=True)
    wc = wc / np.maximum(np.linalg.norm(wc, axis=1, keepdims=True), 1e-9)
    rates = 0.1 * 100.0 ** (np.arange(n_rate) / (n_rate - 1))
    phases = np.arange(n_phase) / n_phase
    n = len(w)
    best = np.full(n, -np.inf)
    br, bp, bm = np.zeros(n), np.zeros(n), np.zeros(n)
    for m in morphs:
        for r in rates:
            bank = lfo_wave(np.full(n_phase, r), phases, np.full(n_phase, m), t, tri)
            bc = bank - bank.mean(1, keepdims=True)
            bc = bc / np.maximum(np.linalg.norm(bc, axis=1, keepdims=True), 1e-9)
            c = wc @ bc.T                                                    # [n,P]
            j = c.argmax(1)
            cm = c[np.arange(n), j]
            upd = cm > best
            best[upd], br[upd], bp[upd], bm[upd] = cm[upd], r, phases[j[upd]], m
    return br, bp, bm


T_LOC = (0.0, 0.375, 0.75, 1.125, 1.5)


def analyze(path):
    D = A.Dump(path)
    d = np.load(path)
    sp_t, sp_p = d["sp_true"], d["sp_pred"]
    tri = dump_tri(d)
    T = D.e_true.shape[-1]
    t = np.minimum(np.arange(T) * HOP + OFF, 1.5 * SR - 1) / SR
    b, k = np.nonzero(D.used & (D.type_true == 0))                    # 使用中の真の LFO
    rate, phase, morph = sp_t[b, k, 0], sp_t[b, k, 1], sp_t[b, k, 2]
    e = D.e_true[b, k]
    const = float(D.e_true[D.used & (D.type_true == 0)].mean())
    has_traj = np.any(D.traj)
    out = dict(n=len(b), rate=rate)
    out["wave_param"] = (D.e_param[b, k], e, const)
    if has_traj:
        w = D.traj[b, k]
        out["wave_traj"] = (w, e, const)
        # [B] 真のレート・morph を固定して、軌跡に最もよく合う位相を探す
        grid = np.arange(128) / 128
        best = np.zeros(len(b))
        best_err = np.full(len(b), np.inf)
        for g in grid:
            err = np.abs(lfo_wave(rate, np.full(len(b), g), morph, t, tri) - w).mean(1)
            upd = err < best_err
            best[upd], best_err[upd] = g, err[upd]
        out["phase_traj"] = (best, phase)
        # [B2] 型紙の当てはめ（真の値を使わない）
        br, bp, bm = template_fit(w, t, tri=tri)
        out["tfit_phase"], out["tfit_rate"], out["tfit_morph"] = (bp, phase), (np.log(br), np.log(rate)), (bm, morph)
        for tc in T_LOC:
            out[f"loc_tfit_{tc}"] = ((bp + br * tc) % 1.0, (phase + rate * tc) % 1.0)
    # 検算: 真の波形に型紙を当てはめれば位相・レートが戻るか
    br0, bp0, _ = template_fit(e, t, tri=tri)
    out["check_phase"], out["check_rate"] = (bp0, phase), (np.log(br0), np.log(rate))
    lfo_p = D.type_pred[b, k] == 0
    out["lfo_pred_frac"] = float(lfo_p.mean())
    out["phase_param"] = (sp_p[b, k, 1][lfo_p], phase[lfo_p], lfo_p)
    # 深さの符号で補正した位相: そのソースの |真の深さ| が最大の列で、予測と真の深さの符号が違えば位相に 0.5 を足す
    sgn_t, sgn_p = np.zeros(len(b)), np.zeros(len(b))
    dest, absd = np.zeros(len(b), int), np.zeros(len(b))
    for j in range(len(b)):
        cols = np.nonzero(D.z_true[b[j]] == k[j])[0]
        c = cols[np.argmax(np.abs(D.d_true[b[j], cols]))]
        sgn_t[j], sgn_p[j] = np.sign(D.d_true[b[j], c]), np.sign(D.d_pred[b[j], c])
        dest[j], absd[j] = c, abs(D.d_true[b[j], c])
    out["dest"], out["absd"] = dest, absd
    flip = (sgn_p != 0) & (sgn_p != sgn_t)
    pc = (sp_p[b, k, 1] + 0.5 * flip) % 1.0
    ok = lfo_p & (sgn_p != 0)
    out["phase_param_signfix"] = (pc[ok], phase[ok], ok)
    out["sign_ok"] = float(np.mean(sgn_p[sgn_p != 0] == sgn_t[sgn_p != 0]))
    out["rate_param"] = (np.log(np.maximum(sp_p[b, k, 0][lfo_p], 1e-6)), np.log(rate[lfo_p]), lfo_p)
    out["morph_param"] = (sp_p[b, k, 2][lfo_p], morph[lfo_p], lfo_p)
    rp, pp = sp_p[b, k, 0], sp_p[b, k, 1]
    for tc in T_LOC:
        out[f"loc_param_{tc}"] = (((pp + rp * tc) % 1.0)[lfo_p], ((phase + rate * tc) % 1.0)[lfo_p], lfo_p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    R = [analyze(p) for p in args.dumps]
    has_traj = "wave_traj" in R[0]

    def agg(fn, key, sel_rate=None, kind="wave"):
        vals = []
        for r in R:
            sel = np.ones(r["n"], bool)
            if callable(sel_rate):
                sel = sel_rate(r)
            elif sel_rate is not None:
                sel = (r["rate"] >= sel_rate[0]) & (r["rate"] < sel_rate[1])
            if kind == "wave":
                w, e, c = r[key]
                vals.append(fn(w[sel], e[sel], c))
            elif kind == "all":
                p, tt = r[key]
                vals.append(fn(p[sel], tt[sel]))
            else:
                p, tt, m = r[key]
                s2 = sel[m]
                vals.append(fn(p[s2], tt[s2]))
        keys = vals[0].keys()
        return {k: (np.mean([v[k] for v in vals]), np.std([v[k] for v in vals], ddof=1) if len(vals) > 1 else 0.0)
                for k in keys}

    f = lambda d, k: f"{d[k][0]:7.3f}±{d[k][1]:.3f}"
    print(f"{len(R)} dumps: {', '.join(args.dumps)}")
    print(f"使用中の真の LFO ソース: 平均 {np.mean([r['n'] for r in R]):.0f} 本 / 予測の型も LFO の割合 "
          f"{np.mean([r['lfo_pred_frac'] for r in R]):.3f}")

    print(f"深さの符号の正解率（そのソースの |深さ| が最大の列、予測がある列）: {np.mean([r['sign_ok'] for r in R]):.3f}")
    print("\n[A] 波形の再現（使用中の真の LFO すべて）   相関（中央値）  |相関|（上下反転も一致）  振れ幅の比    平均絶対誤差   定数予測と比べた改善")
    rows = [("パラメータから作った e(n)", "wave_param")] + ([("Stage 1 の軌跡", "wave_traj")] if has_traj else [])
    for label, key in rows:
        for rb in [None] + RATE_BINS:
            s = agg(wave_stats, key, rb)
            name = label if rb is None else f"    レート {rb[0]:>4}〜{rb[1]:<5.0f}Hz"
            print(f"  {name:24s}{f(s, 'corr'):>16s}{f(s, 'abscorr'):>18s}{f(s, 'amp'):>16s}{f(s, 'mae'):>16s}{f(s, 'skill'):>16s}")

    print("\n[B][C] 位相・レート・morph              一致度 / 相関   半周期ずれを同一視   ばらつき比       誤差の改善")
    items = []
    if has_traj:
        items += [("位相（軌跡に最もよく合う位相、真のレート・morph）", circ_stats, "phase_traj", "all"),
                  ("位相（型紙の当てはめ、真の値なし）", circ_stats, "tfit_phase", "all"),
                  ("レート（型紙の当てはめ、対数）", lin_stats, "tfit_rate", "all"),
                  ("morph（型紙の当てはめ）", lin_stats, "tfit_morph", "all")]
    items += [("位相（検算: 真の波形に型紙を当てはめ）", circ_stats, "check_phase", "all"),
              ("レート（検算: 真の波形に型紙を当てはめ）", lin_stats, "check_rate", "all")]
    items += [("位相（パラメータ）", circ_stats, "phase_param", "param"),
              ("位相（パラメータ、深さの符号で補正）", circ_stats, "phase_param_signfix", "param"),
              ("レート（パラメータ、対数）", lin_stats, "rate_param", "param"),
              ("morph（パラメータ）", lin_stats, "morph_param", "param")]
    for label, fn, key, kind in items:
        for rb in [None] + RATE_BINS:
            s = agg(fn, key, rb, kind=kind)
            name = label if rb is None else f"    レート {rb[0]:>4}〜{rb[1]:<5.0f}Hz"
            half = f(s, "agree_half") if "agree_half" in s else f"{'—':>13s}"
            print(f"  {name:34s}{f(s, 'agree'):>15s}{half:>18s}{f(s, 'ratio'):>16s}{f(s, 'skill'):>16s}")

    print("\n[D]時刻 t での位相（φ + レート × t）の一致度")
    print(f"  {'':34s}" + "".join(f"{'t=' + str(tc) + 's':>16s}" for tc in T_LOC))
    srcs = ([("型紙の当てはめ", "loc_tfit", "all")] if has_traj else []) + [("パラメータ", "loc_param", "param")]
    for label, key, kind in srcs:
        for rb in [None] + RATE_BINS:
            name = label if rb is None else f"    レート {rb[0]:>4}〜{rb[1]:<5.0f}Hz"
            vals = [agg(circ_stats, f"{key}_{tc}", rb, kind=kind) for tc in T_LOC]
            print(f"  {name:34s}" + "".join(f"{f(v, 'agree'):>16s}" for v in vals))

    names = [d[0] for d in ds.dest_specs(SR)]
    cols = ([("真レート固定", "phase_traj", "all"), ("型紙", "tfit_phase", "all")] if has_traj else []) \
        + [("パラメータ", "phase_param", "param")]
    print("\n[E]変調先・|深さ|ごとの位相の一致度（レート 1 Hz 以上、3 つの dump で数を合計）")
    print(f"  {'':16s}{'本数':>8s}" + "".join(f"{c[0]:>14s}" for c in cols))
    groups = [(nm, lambda r, m=m: (r["dest"] == m) & (r["rate"] >= 1.0)) for m, nm in enumerate(names)]
    groups += [(f"|深さ| {lo}〜{hi}", lambda r, lo=lo, hi=hi: (r["absd"] >= lo) & (r["absd"] < hi) & (r["rate"] >= 1.0))
               for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 1.01))]
    for nm, fn_sel in groups:
        cnt = sum(int(fn_sel(r).sum()) for r in R)
        if cnt < 60:
            continue
        vals = [agg(circ_stats, key, fn_sel, kind=kind)["agree"][0] for _, key, kind in cols]
        print(f"  {nm:16s}{cnt:8d}" + "".join(f"{v:14.3f}" for v in vals))


if __name__ == "__main__":
    main()
