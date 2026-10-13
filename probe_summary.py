r"""probe_summary.py — 実験 37（phase_probe.py）の結果の一覧

  python probe_summary.py probe/            # probe/*.json をまとめ、同じ seed 0 の基準（型紙・Stage 2）と並べる
"""
import glob
import json
import os
import sys

import numpy as np

import lfo_phase_check as L

BINS = ["all", "1.0-3", "3.0-10"]
BIN_RANGE = {"all": (0.0, 99.0), "1.0-3": (1.0, 3.0), "3.0-10": (3.0, 10.01)}


def baselines(test_dump):
    """同じ test の軌跡での基準: 真のレートを与えた位相の探索、型紙の当てはめ、実際の Stage 2 の位相（全 LFO）"""
    r = L.analyze(test_dump)
    d = np.load(test_dump)
    b, k = np.nonzero(d["used"] & (d["type_true"] == 0))
    p_all = d["sp_pred"][b, k, 1]
    rows = {}
    for name, (p, t) in [("真のレートを与えた探索", r["phase_traj"]), ("型紙の当てはめ", r["tfit_phase"]),
                         ("実際の Stage 2（全 LFO）", (p_all, r["phase_traj"][1]))]:
        out = {}
        for bn in BINS:
            lo, hi = BIN_RANGE[bn]
            sel = (r["rate"] >= lo) & (r["rate"] < hi)
            out[bn] = L.circ_stats(p[sel], t[sel])
        rows[name] = out
    return rows


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "probe"
    R = {}
    for f in sorted(glob.glob(os.path.join(root, "*.json"))):
        j = json.load(open(f))
        a = j["args"]
        R[(a["train"], a["arch"], a["loss"])] = j
    base = baselines(next(iter(R.values()))["args"]["test_dump"])
    print("基準（test の実際の軌跡、seed 0）                 位相の一致度: 全体 / 1〜3 Hz / 3〜10 Hz")
    for name, out in base.items():
        print(f"  {name:28s}" + "".join(f"{out[bn]['agree']:9.3f}" for bn in BINS))

    hdr = (f"{'':22s}{'きれいな波形で評価':^30s}{'実際の軌跡で評価':^30s}{'ばらつき比':>10s}{'レートの相関':>12s}"
           f"{'学習の歩数':>10s}")
    for tr, label in (("clean", "きれいな波形で学習"), ("synth", "崩した波形で学習"), ("traj", "実際の軌跡で学習")):
        print(f"\n[{label}]  位相の一致度: 全体 / 1〜3 Hz / 3〜10 Hz")
        print(hdr)
        for arch in ("pool", "ends", "flat"):
            for loss in ("circ2", "cs", "wave"):
                j = R.get((tr, arch, loss))
                if j is None:
                    continue
                c, w = j["test_clean"], j["test_traj"]
                ev = "test_clean" if tr == "clean" else "test_traj"
                best_step = max(j["history"], key=lambda h: h["val_phase_agree"])["step"]
                print(f"  {arch + ' / ' + loss:20s}" + "".join(f"{c[bn]['phase_agree']:10.3f}" for bn in BINS)
                      + "".join(f"{w[bn]['phase_agree']:10.3f}" for bn in BINS)
                      + f"{j[ev]['all']['phase_ratio']:10.2f}{j[ev]['all']['rate_corr']:12.3f}{best_step:10d}")


if __name__ == "__main__":
    main()
