r"""traj_slot_check.py — Stage 1 の軌跡が崩れる原因の切り分け（実験 39）

  python traj_slot_check.py dump_v22s_two_s{0,1,2}_test.npz

使用中の真の LFO ソースの枠 k ごとに、軌跡 traj[k] と真の波形の時間方向の相関を測る。
  自分の枠    : corr(traj[k], e[k])
  最良の枠    : 使用中の真のソースすべて j のうち corr(traj[k], e[j]) が最大のもの
                → 最良が別の枠なら、番号の取り違え（別のソースの波形を出している）
  上下反転    : corr(traj[k], e[k]) ≤ −0.3 の割合（深さの符号と入れ替わって反転した波形を出している）
これを、レート帯、音の中の使用中のソースの数、使用中の LFO の数で分ける。

成分の分解（3〜10 Hz の LFO）: 約 0.33 秒（21 コマ）の移動平均を「ゆっくりした成分」、残りを「速い揺れ」とする。
3 Hz 以上の LFO は 0.33 秒で 1 周期以上あるので、真の波形のゆっくりした成分はほぼ 0。
  速い揺れの相関・振れ幅の比 : 軌跡の速い揺れを真の波形の速い揺れと比べる
  速い揺れの割合            : 軌跡の分散のうち速い揺れが占める割合（残りは真の LFO にない、ゆっくりした成分）
ENV の枠: 使用中の真の ENV の枠で、軌跡と真の波形の相関と平均絶対誤差（中央値）
"""
import argparse

import numpy as np


def corr_rows(a, b):
    ac, bc = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    return (ac * bc).sum(-1) / np.maximum(np.sqrt((ac ** 2).sum(-1) * (bc ** 2).sum(-1)), 1e-12)


def smooth(x, w=21):
    """時間方向の移動平均（端は端の値で延長）。x: [n,T]"""
    k = np.ones(w) / w
    pad = np.pad(x, ((0, 0), (w // 2, w - 1 - w // 2)), mode="edge")
    return np.stack([np.convolve(r, k, mode="valid") for r in pad])


def analyze(path):
    d = np.load(path)
    used, tt = d["used"], d["type_true"]
    traj, e = d["traj"], d["e_true"]
    sp = d["sp_true"]
    rng = np.random.default_rng(0)
    bl, kl = np.nonzero(used & (tt == 0))
    w, et = traj[bl, kl], e[bl, kl]
    ws, es = smooth(w), smooth(et)
    wh, eh = w - ws, et - es
    hp_corr = corr_rows(wh, eh)
    hp_amp = wh.std(1) / np.maximum(eh.std(1), 1e-9)
    fast_share = wh.var(1) / np.maximum(w.var(1), 1e-12)
    slow_std = ws.std(1)
    be, ke = np.nonzero(used & (tt == 1))
    env = dict(corr=float(np.median(corr_rows(traj[be, ke], e[be, ke]))),
               mae=float(np.median(np.abs(traj[be, ke] - e[be, ke]).mean(1))))
    rows = []
    for i, (b, k) in enumerate(zip(bl, kl)):
        js = np.nonzero(used[b])[0]
        c = np.array([corr_rows(traj[b, k], e[b, j]) for j in js])
        self_c = c[list(js).index(k)]
        best = js[np.argmax(c)]
        # 偶然の水準: 別の音（同じ数の使用中ソースを持つもの）の真の波形と比べたときの最大の相関
        while True:
            b2 = rng.integers(len(used))
            if b2 != b and used[b2].sum() == len(js):
                break
        c_null = max(corr_rows(traj[b, k], e[b2, j]) for j in np.nonzero(used[b2])[0])
        rows.append(dict(self=self_c, best_other=(best != k) and (c.max() > self_c + 0.1),
                         best=c.max(), null=c_null, best_env=(best != k) and tt[b, best] == 1,
                         inv=self_c <= -0.3, n_used=len(js),
                         n_lfo=int((used[b] & (tt[b] == 0)).sum()), rate=sp[b, k, 0],
                         hp_corr=hp_corr[i], hp_amp=hp_amp[i], fast_share=fast_share[i], slow_std=slow_std[i]))
    return rows, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    A = [analyze(p) for p in args.dumps]
    R = [a[0] for a in A]

    def table(title, key_fn, keys):
        print(f"\n{title}")
        print(f"  {'':18s}{'本数':>7s}{'自分の枠の相関':>12s}{'最良の枠の相関':>12s}{'偶然の水準':>10s}"
              f"{'別の枠が最良':>10s}{'うち ENV':>9s}{'上下反転':>9s}")
        for kname, kval in keys:
            vals = []
            for rows in R:
                sel = [r for r in rows if key_fn(r) == kval]
                if not sel:
                    continue
                vals.append((len(sel), np.median([r["self"] for r in sel]), np.median([r["best"] for r in sel]),
                             np.median([r["null"] for r in sel]),
                             np.mean([r["best_other"] for r in sel]), np.mean([r["best_env"] for r in sel]),
                             np.mean([r["inv"] for r in sel])))
            if not vals:
                continue
            v = np.array(vals)
            print(f"  {kname:18s}{int(v[:, 0].sum()):7d}" + "".join(f"{x:12.3f}" for x in v[:, 1:4].mean(0))
                  + "".join(f"{x:10.3f}" for x in v[:, 4:].mean(0)))

    table("全体", lambda r: 0, [("すべて", 0)])
    table("レート帯", lambda r: 0 if r["rate"] < 1 else (1 if r["rate"] < 3 else 2),
          [("0.1〜1 Hz", 0), ("1〜3 Hz", 1), ("3〜10 Hz", 2)])
    table("使用中のソースの数", lambda r: r["n_used"], [(f"{n} 本", n) for n in (1, 2, 3, 4)])
    table("使用中の LFO の数", lambda r: r["n_lfo"], [(f"{n} 本", n) for n in (1, 2, 3, 4)])

    def decomp(title, key_fn, keys):
        print(f"\n{title}")
        print(f"  {'':18s}{'本数':>7s}{'速い揺れの相関':>12s}{'速い揺れの振れ幅の比':>14s}{'速い揺れの割合':>12s}"
              f"{'ゆっくりした成分の標準偏差':>16s}")
        for kname, kval in keys:
            vals = []
            for rows in R:
                sel = [r for r in rows if r["rate"] >= 3.0 and key_fn(r) == kval]
                if not sel:
                    continue
                vals.append([len(sel)] + [np.median([r[c] for r in sel])
                                          for c in ("hp_corr", "hp_amp", "fast_share", "slow_std")])
            if not vals:
                continue
            v = np.array(vals)
            print(f"  {kname:18s}{int(v[:, 0].sum()):7d}{v[:, 1].mean():12.3f}{v[:, 2].mean():14.3f}"
                  f"{v[:, 3].mean():12.3f}{v[:, 4].mean():16.3f}")

    print("\n[成分の分解] 3〜10 Hz の LFO（真の LFO 波形の標準偏差は約 0.34）")
    decomp("全体", lambda r: 0, [("すべて", 0)])
    decomp("使用中の LFO の数", lambda r: r["n_lfo"], [(f"{n} 本", n) for n in (1, 2, 3)])
    decomp("使用中のソースの数", lambda r: r["n_used"], [(f"{n} 本", n) for n in (2, 3, 4)])

    ec = np.array([[a[1]["corr"], a[1]["mae"]] for a in A])
    print(f"\n[ENV の枠] 軌跡と真の波形の相関 {ec[:, 0].mean():.3f}±{ec[:, 0].std(ddof=1) if len(ec) > 1 else 0:.3f}、"
          f"平均絶対誤差 {ec[:, 1].mean():.3f}")


if __name__ == "__main__":
    main()
