r"""check_typed_layout.py — ENV 4 枠 ＋ LFO 3 枠への並べ替え（実験 41）の検算

  MOD_LAYOUT=typed python check_typed_layout.py
  python check_typed_layout.py                     # 比較用: 並べ替えない 4 枠（合成し直しの誤差の基準）

1. 並べ替えた 7 枠のパラメータ・R で音を合成し直し、保存されている音と一致するか（int16 の量子化の範囲）
2. 型ごとの使用中の本数・枠の番号の範囲・正準順序（使用中を先頭、ENV は attack 順、LFO はレート順）
"""
import os

import numpy as np

import diff_synth_tf as ds
from diff_synth_fast import synth_fast
from train import Data

D = Data("dataset_v7")
print(f"枠の並び: {ds.LAYOUT}（K = {ds.K_ENVS}）")
SR, N, T_OFF = D.man["sr"], D.man["N"], D.man["t_off"]

# 1. 合成し直して比べる（MIDI ごとに倍音数が違うので、MIDI ごとに 16 件ずつ）
#    保存された音とは、合成し直しの誤差（4 枠のままでも同じだけ出る）があるので、
#    両方の並びで合成し直した音どうしも比べる（/tmp/resyn_<並び>.npy に保存し、もう一方があれば比べる）
worst, ys = 0.0, []
for m_val in (36, 60, 84):
    idx = D.splits["test"][D.P["midi"][D.splits["test"]] == m_val][:16]
    x, b = D.batch(idx)
    fmax = float(b["f0"][0]) * 2 ** (100 / 1200)
    nh = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    y = synth_fast(b["theta_stat"], b["theta_osc"], b["gmk"], b["src_params"], b["R"], b["f0"],
                   T_OFF, N, SR, nh[0], nh[1], src_type=b["src_type"]).numpy()
    err = np.abs(np.clip(y, -32768 / D.scale, 32767 / D.scale) - x.numpy()).max()
    worst = max(worst, err)
    ys.append(y)
    print(f"MIDI {m_val}: 合成し直した音と保存された音の差の最大 {err:.2e}（量子化の刻み {1 / D.scale:.2e}）")
np.save(f"/tmp/resyn_{ds.LAYOUT}.npy", np.concatenate(ys))
other = "/tmp/resyn_" + ("mixed" if ds.LAYOUT == "typed" else "typed") + ".npy"
if os.path.exists(other):
    print(f"4 枠と 7 枠で合成し直した音どうしの差の最大: {np.abs(np.load(other) - np.concatenate(ys)).max():.2e}")

if ds.LAYOUT != "typed":
    raise SystemExit(0)

# 2. 枠の中身
st, sp, z, R = D.P["src_type"], D.P["src_params"], D.P["z"].astype(int), D.P["R"]
used = np.stack([(z == k).any(1) for k in range(ds.K_ENVS)], 1)
print("枠の型:", st[0].tolist(), " 全件同じ:", bool((st == st[0]).all()))
print("使用中の ENV の本数の分布:", np.bincount(used[:, :4].sum(1)).tolist(),
      " LFO の本数の分布:", np.bincount(used[:, 4:].sum(1)).tolist())
print("使用中の枠が先頭に詰まっているか: ENV", bool((np.diff(used[:, :4].astype(int), axis=1) <= 0).all()),
      " LFO", bool((np.diff(used[:, 4:].astype(int), axis=1) <= 0).all()))
both = used[:, :4][:, 1:] & used[:, :4][:, :-1]
print("ENV の attack が昇順か:", bool((np.diff(sp[:, :4, 0], axis=1)[both] >= 0).all()))
both = used[:, 4:][:, 1:] & used[:, 4:][:, :-1]
print("LFO のレートが昇順か:", bool((np.diff(sp[:, 4:, 0], axis=1)[both] >= 0).all()))
print("R が使われていない枠で 0 か:", bool((np.abs(R[~used]) == 0).all()))
print(f"保存された音との差の最大 {worst:.2e}（4 枠のままで合成し直しても同じだけ出る。上の「音どうしの差」で判断する）")
