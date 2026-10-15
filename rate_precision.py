r"""rate_precision.py — LFO のレートの予測の相対誤差（実験 42）

  python rate_precision.py        # dump_v25_single_s*、dump_v27_single_s* を比べる

±5% / ±10% 以内の割合と相対誤差の中央値を、使用中の LFO すべて、LFO が 1 本の音の 3〜10 Hz、
LFO が 1 本でカットオフを |深さ| 0.5 以上で変調する音（変調スペクトルの山で 79% が ±10% 以内に入る簡単な音）で出す
"""
import numpy as np

def stats(rel):
    return f"±5% 以内 {np.mean(rel <= 0.05):.2f}、±10% 以内 {np.mean(rel <= 0.1):.2f}、相対誤差の中央値 {np.median(rel):.3f}（{len(rel)} 本）"

for tag in ("v25", "v27"):
    out = {"all": [], "easy_cutoff": [], "one_lfo_fast": []}
    for s in range(3):
        d = np.load(f"dump_{tag}_single_s{s}_test.npz")
        used, z, dt = d["used"], d["z_true"].astype(int), d["d_true"]
        rt, rp = d["sp_true"][..., 0], d["sp_pred"][..., 0]
        n_lfo = used[:, 4:].sum(1)
        b, k = np.nonzero(used[:, 4:]); k = k + 4
        rel = np.abs(rp[b, k] - rt[b, k]) / rt[b, k]
        out["all"].append(rel)
        fast = (rt[b, k] >= 3) & (n_lfo[b] == 1)
        out["one_lfo_fast"].append(rel[fast])
        easy = []
        for j in range(len(b)):
            if n_lfo[b[j]] != 1 or not (1 <= rt[b[j], k[j]] <= 10):
                continue
            if z[b[j], 4] == k[j] and abs(dt[b[j], 4]) >= 0.5:
                easy.append(rel[j])
        out["easy_cutoff"].append(np.array(easy))
    for key, name in (("all", "使用中の LFO すべて"), ("one_lfo_fast", "LFO 1 本の音、3〜10 Hz"),
                      ("easy_cutoff", "LFO 1 本でカットオフを深さ 0.5 以上（1〜10 Hz）")):
        print(f"{tag} {name}: {stats(np.concatenate(out[key]))}")
