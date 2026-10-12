# jtfs_local_select.npz から物差しを決める: attack/release の比 0.8〜1.25 かつ D* に等しい音量差 0.5〜1.0 dB のうち、比が最も 1 に近いもの
import numpy as np, sys
d = np.load("jtfs_local_select.npz"); kind, mag = d["kind"], d["mag"]
DBS = [0.25, 0.5, 1.0, 2.0]
best = None
for k in [k for k in d.files if "/" in k]:
    v = d[k]
    da, dr = np.median(v[kind == "attack"]), np.median(v[kind == "release"])
    Ds = np.median(v[np.isin(kind, ["attack", "release"])])
    g = np.maximum.accumulate([np.median(v[(kind == "gain") & (mag == m)]) for m in DBS])
    if g[0] >= Ds or g[-1] < Ds:
        continue
    i = np.argmax(g >= Ds); db = float(np.exp(np.interp(Ds, g[i-1:i+1], np.log(DBS[i-1:i+1]))))
    r = da / dr
    print(f"{k:18s} 比 {r:.3f} 音量差 {db:.2f} dB", file=sys.stderr)
    if 0.8 <= r <= 1.25 and 0.5 <= db <= 1.0 and (best is None or abs(np.log(r)) < best[0]):
        best = (abs(np.log(r)), k)
print(best[1] if best else "u2/P32ms/30dB/max")
