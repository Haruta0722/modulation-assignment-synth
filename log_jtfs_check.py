r"""log_jtfs_check.py — JTFS の係数に対数をかけた距離で、3 種類の弁別閾が同じ距離にそろうかを確かめる

  python log_jtfs_check.py --n 300

同じ音の組（音量の列にだけつながる ENV、sustain ≥ 0.8、深さ 1・静的な値 0。calibrate_volume.py の d1 と同じ）で、
  時間の形: 音量の ENV の attack / release を秒で ×0.8 / ×1.25（弁別閾 ≈ 25%）
  音量    : 音全体を ±0.25, 0.5, 1, 2 dB（音量の弁別閾はおおよそ 0.5〜1 dB）
  音程    : 基本周波数 f0 を ±2, 5, 10, 20 セント（純粋な音程のずれ。fin の制御値を動かすと smooth_clamp の
            傾きの違いで 2 つのオシレータの音程差まで変わり、うなりがずれるので使わない）
の距離を測る。距離は JTFS（PNP の設定、折り返しを防ぐ）について
  lin    : ‖Φ1 − Φ0‖ / ‖Φ0‖（これまでの相対距離）
  log(c) : sqrt(mean((log(Φ1 + ε) − log(Φ0 + ε))²))、ε = c × mean(Φ0)（c が小さいほど対数が強くかかる）
  logmel : 対数メル（モデルの入力）のフレームごとの平均絶対差
  u1_t(c), u2_t(c): JTFS の 1 次・2 次を時間方向に平均せず、対数（ε = c × 平均）をかけてフレームごとに比べた平均絶対差
各距離で D* = attack / release ±25% の距離の中央値とし、距離の中央値が D* に等しくなる音量差（dB）と
音程差（セント）を、測った点の間を補間して求める。
"""
import argparse, os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import train as T
from audibility import logmel, quantize
from diff_synth_fast import synth_fast
from jtfs_tf import JTFS

K = ds.K_ENVS
LEVEL = {0, 1}
FIN1, FIN2 = 2, 3
DBS = [0.25, 0.5, 1.0, 2.0]
CENTS = [2, 5, 10, 20]
FLOORS = [30, 40]
SMOOTH_MS = [16, 32, 64]
GAUSS_MS = [32, 64, 128]                             # パワーを平滑化するガウス窓の標準偏差（JTFS の局所平均 T にあたる）
BOXCAR = False                                       # 振幅の移動平均（前の版）も出すか


def erb(f):
    return 24.7 * (4.37 * f / 1000 + 1)


def roex_bank(sr, n_fft, n_bands=64, f_lo=50.0, f_hi=7000.0):
    """ERB の間隔で並べた roex(p) 聴覚フィルタ（Glasberg & Moore 1990）。パワーに掛ける重み [n_bins, n_bands]
    W(g) = (1 + p g) exp(−p g), g = |f − fc| / fc, p = 4 fc / ERB(fc)。裾が dB でほぼ直線に落ちる"""
    e_lo, e_hi = [21.4 * np.log10(4.37 * f / 1000 + 1) for f in (f_lo, f_hi)]
    fc = (10 ** (np.linspace(e_lo, e_hi, n_bands) / 21.4) - 1) * 1000 / 4.37
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    g = np.abs(f[:, None] - fc[None, :]) / fc[None, :]
    p = 4 * fc / erb(fc)
    return ((1 + p * g) * np.exp(-p * g)).astype(np.float32)


def gauss_smooth(a, sigma):
    """最後の軸（時間）をガウス窓（標準偏差 sigma フレーム）で平滑化（移動平均より漏れが少ない）"""
    if sigma <= 0:
        return a
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * (a.ndim - 1) + [(r, r)]
    ap = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += w * ap[..., i:i + a.shape[-1]]
    return out


def smooth(a, w):
    """最後の軸（時間）を幅 w の移動平均で平滑化（端は同じ長さを保つ）"""
    if w <= 1:
        return a
    k = np.ones(w) / w
    pad = [(0, 0)] * (a.ndim - 1) + [(w // 2, w - 1 - w // 2)]
    ap = np.pad(a, pad, mode="edge")
    c = np.cumsum(ap, axis=-1)
    c = np.concatenate([np.zeros(c.shape[:-1] + (1,)), c], axis=-1)
    return (c[..., w:] - c[..., :-w]) / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v7")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no_chorus", action="store_true",
                    help="コーラスを素通しにする（20 ms の遅延による櫛形フィルタが音程差を音色差に変えるかの確認用）")
    ap.add_argument("--cents", default="", help="音程のずらし量（例: 0.1,0.5,2,20）。--cents none で音程を測らない")
    ap.add_argument("--reps", default="mel,u1,u2,ex", help="測る表現（mel, u1, u2, ex）")
    ap.add_argument("--gauss_ms", default="", help="パワーの平滑化のガウス窓（例: 32,64）")
    ap.add_argument("--floors", default="", help="下限（例: 30,40）")
    ap.add_argument("--out", default="", help="1 組ごとの距離を保存する .npz")
    args = ap.parse_args()
    global CENTS, GAUSS_MS, FLOORS, REPS
    if args.cents == "none":
        CENTS = []
    elif args.cents:
        CENTS = [float(v) for v in args.cents.split(",")]
    if args.gauss_ms:
        GAUSS_MS = [int(v) for v in args.gauss_ms.split(",")]
    if args.floors:
        FLOORS = [int(v) for v in args.floors.split(",")]
    REPS = args.reps.split(",")
    if args.no_chorus:
        import diff_synth_fast
        diff_synth_fast.chorus_fast = lambda x, rate, depth, feedback, sr: x
    rng = np.random.default_rng(0)

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    TAIL = N / SR - T_OFF
    phi = JTFS(sr=SR, J=8, Q=6, J_fr=2, Q_fr=2, antialias=True)
    feat = tf.function(phi.call, reduce_retracing=True)

    P = data.P
    picks = []
    for i in np.concatenate([data.splits["test"], data.splits["val"]]):
        z, sp, st = P["z"][i].astype(int), P["src_params"][i], P["src_type"][i]
        for k in range(K):
            cols = set(np.nonzero(z == k)[0])
            if cols and st[k] == 1 and cols <= LEVEL and sp[k, 3] >= 0.8:
                picks.append((i, k))
                break
        if len(picks) >= args.n:
            break

    def synth(b, sp, R, th, f0_scale=None):
        """f0_scale を渡すと基本周波数そのものを倍率で変える（2 つのオシレータの音程差は変わらない純粋な音程のずれ）"""
        f0 = b["f0"].numpy()
        f0s = f0 * (1.0 if f0_scale is None else f0_scale)
        out = np.zeros((len(f0), N), np.float32)
        for f in np.unique(f0):
            s = np.nonzero(f0 == f)[0]
            fmax = float(f) * 2 ** (120 / 1200)            # 倍音数は元の f0 から（ずらしてもナイキストを超えない余裕）
            hs, hq = int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2)
            g = lambda t: tf.gather(tf.constant(t) if isinstance(t, np.ndarray) else t, s)
            out[s] = synth_fast(g(th), g(b["theta_osc"]), g(b["gmk"]), g(sp), g(R),
                                tf.constant(f0s[s].astype(np.float32)),
                                T_OFF, N, SR, hs, hq, src_type=g(b["src_type"])).numpy()
        return out

    @tf.function(reduce_retracing=True)
    def resolved(x):
        """時間方向に平均しない表現: 1 次のスカログラム U1 [B,F,T]、2 次の U2 [B,C,F,T]、対数メル [B,T,64]"""
        u1 = phi.scalogram(x)
        u2 = [u for psi_t, _ in phi.f2_t for psi_f, _ in phi.f2_f for u in phi._conv2d_c(u1, psi_t, psi_f)]
        return u1, tf.stack(u2, axis=1), logmel(x, SR)

    W_ex = tf.constant(roex_bank(SR, 1024))

    @tf.function(reduce_retracing=True)
    def excitation(x):
        """興奮パターン: STFT（窓 32 ms・ホップ 4 ms）のパワーに roex 聴覚フィルタをかける [B,bands,T]"""
        S = tf.abs(tf.signal.stft(x, 512, 64, fft_length=1024)) ** 2
        return tf.transpose(tf.matmul(S, W_ex), [0, 2, 1])

    def phi_of(x):
        # 聞こえ方の評価なので量子化しない（無音付近の量子化雑音が対数で強調されるのを避ける）
        xt = tf.constant(x)
        u1, u2, lm = resolved(xt)
        F = dict(phi=feat(xt).numpy(), u1=u1.numpy(), u2=u2.numpy())
        if "mel" in REPS:
            F["mel"] = np.maximum(np.exp(lm.numpy()) - 1e-5, 0.0).transpose(0, 2, 1)   # [B,64,T] 振幅
        if "ex" in REPS:
            F["ex"] = excitation(xt).numpy()
        return F

    def take(F, ok):
        return {k: v[ok] for k, v in F.items()}

    def dists(F0, F1):
        a0, a1 = F0["phi"], F1["phi"]
        out = {"lin": np.linalg.norm(a1 - a0, axis=1) / np.maximum(np.linalg.norm(a0, axis=1), 1e-12)}
        # 時間分解能のある表現: 振幅を時間方向に W ms で移動平均（耳の時間的な積分の代わり）→ 最大値から L dB 下を
        # 下限にして対数 → 周波数（とチャネル）で平均したフレームごとの差を、時間方向に 上位 10% / 最大 でまとめる
        hop_ms = {"mel": 16.0, "u1": 8.0, "u2": 8.0}
        # 興奮パターン（パワー）: 時間方向に平滑化 → dB → 「どこか 1 か所で何 dB 変わったか」（Zwicker 型の判定）
        for sg in ((8, 16) if "ex" in REPS else ()):
            e0 = gauss_smooth(F0["ex"], sg / 4.0)
            e1 = gauss_smooth(F1["ex"], sg / 4.0)
            peak = e0.max(axis=(1, 2), keepdims=True)
            for L in (40, 60):
                fl = peak * 10 ** (-L / 10)
                dd = np.abs(10 * np.log10((e1 + fl) / (e0 + fl)))                    # [B,bands,T] dB
                out[f"ex/P{sg}ms/{L}dB/max"] = dd.max((1, 2))
                out[f"ex/P{sg}ms/{L}dB/p99"] = np.percentile(dd.reshape(len(dd), -1), 99, axis=1)
                out[f"ex/P{sg}ms/{L}dB/fmean"] = dd.mean(1).max(1)
        # パワー（振幅の 2 乗）をガウス窓で平滑化 → 干渉の成分が打ち消し合う。振幅の次元に戻して比べる
        for name in [r for r in ("mel", "u1", "u2") if r in REPS]:
            for sg in GAUSS_MS:
                s = sg / hop_ms[name]
                b0 = np.sqrt(gauss_smooth(F0[name] ** 2, s))
                b1 = np.sqrt(gauss_smooth(F1[name] ** 2, s))
                fax = tuple(range(1, b0.ndim - 1))
                peak = b0.max(axis=tuple(range(1, b0.ndim)), keepdims=True)
                for L in FLOORS:
                    e = peak * 10 ** (-L / 20)
                    dt = np.abs(np.log(b1 + e) - np.log(b0 + e)).mean(fax)
                    out[f"{name}/P{sg}ms/{L}dB/p90"] = np.percentile(dt, 90, axis=1)
                    out[f"{name}/P{sg}ms/{L}dB/max"] = dt.max(1)
        if not BOXCAR:
            return out
        for name in ("mel", "u1", "u2"):
            for W in SMOOTH_MS:
                w = max(int(round(W / hop_ms[name])), 1)
                b0, b1 = smooth(F0[name], w), smooth(F1[name], w)
                fax = tuple(range(1, b0.ndim - 1))
                peak = b0.max(axis=tuple(range(1, b0.ndim)), keepdims=True)
                for L in FLOORS:
                    e = peak * 10 ** (-L / 20)
                    dt = np.abs(np.log(b1 + e) - np.log(b0 + e)).mean(fax)            # [B,T]
                    out[f"{name}/{W}ms/{L}dB/p90"] = np.percentile(dt, 90, axis=1)
                    out[f"{name}/{W}ms/{L}dB/max"] = dt.max(1)
        return out

    rows = []                                        # (種類, 大きさ, 距離 dict の 1 行)
    for s0 in range(0, len(picks), args.batch):
        chunk = picks[s0:s0 + args.batch]
        idx = np.array([p[0] for p in chunk]); ks = np.array([p[1] for p in chunk])
        x, b = data.batch(idx)
        sp0, R0, th0 = b["src_params"].numpy(), b["R"].numpy(), b["theta_stat"].numpy()
        z = b["z"].numpy(); B = len(idx)
        R1, th1 = R0.copy(), th0.copy()
        for j in range(B):
            for c in np.nonzero(z[j] == ks[j])[0]:
                R1[j, ks[j], c], th1[j, c] = 1.0, 0.0
        x0 = synth(b, sp0, R1, th1)
        F0 = phi_of(x0)

        def add(kind, mag, F1, ok):
            d = dists(take(F0, ok), take(F1, ok))
            for j in range(ok.sum()):
                rows.append((kind, mag, {k: float(v[j]) for k, v in d.items()}))

        for param, col in (("attack", 0), ("release", 4)):
            for f in (0.8, 1.25):
                sp = sp0.copy(); ok = np.zeros(B, bool)
                for j in range(B):
                    a, h, d, s, r = sp0[j, ks[j]]
                    if param == "attack" and a >= 0.08 and a * f + h + d <= T_OFF:
                        sp[j, ks[j], 0], ok[j] = a * f, True
                    if param == "release" and r >= 0.08 and r * f <= TAIL:
                        sp[j, ks[j], 4], ok[j] = r * f, True
                if ok.any():
                    add(param, f, phi_of(synth(b, sp, R1, th1)), ok)
        for db in DBS:
            sgn = np.where(rng.random(B) < 0.5, 1.0, -1.0)
            add("gain", db, phi_of((x0 * (10 ** (sgn * db / 20))[:, None]).astype(np.float32)), np.ones(B, bool))
        for c in CENTS:
            sgn = np.where(rng.random(B) < 0.5, 1.0, -1.0)
            add("pitch", c, phi_of(synth(b, sp0, R1, th1, f0_scale=2 ** (sgn * c / 1200))), np.ones(B, bool))
        print(f"  {s0 + B} / {len(picks)}", flush=True)

    kind = np.array([r[0] for r in rows]); mag = np.array([r[1] for r in rows], float)
    names = list(rows[0][2])

    if args.out:
        np.savez(args.out, kind=kind, mag=mag, **{k: np.array([r[2][k] for r in rows]) for k in names})

    def crossing(vals, mags, target):
        """距離の中央値が target に等しくなる大きさ（対数軸で補間）"""
        if len(vals) == 0:
            return "-"
        v = np.maximum.accumulate(np.array(vals))
        if v[0] >= target:
            return f"<{mags[0]:g}"
        if v[-1] < target:
            return f">{mags[-1]:g}"
        i = np.argmax(v >= target)
        lm = np.interp(target, v[i - 1:i + 1], np.log(mags[i - 1:i + 1]))
        return f"{np.exp(lm):.2g}"

    print(f"\n対象の音 {len(picks)}（attack {np.sum(kind == 'attack')} / release {np.sum(kind == 'release')} 組）")
    print(f"{'距離':18s}{'attack':>9s}{'release':>9s}{'比':>6s}{'音量差 dB':>11s}{'音程差 セント':>14s}   音量 {DBS} dB の距離 / 音程 {CENTS} セントの距離")
    good = []
    for k in names:
        v = np.array([r[2][k] for r in rows])
        da, dr = np.median(v[kind == "attack"]), np.median(v[kind == "release"])
        Dstar = np.median(v[np.isin(kind, ["attack", "release"])])
        g = [np.median(v[(kind == "gain") & (mag == m)]) for m in DBS]
        p = [np.median(v[(kind == "pitch") & (mag == m)]) for m in CENTS]
        cg, cp = crossing(g, np.array(DBS), Dstar), crossing(p, np.array(CENTS, float), Dstar)
        print(f"{k:18s}{da:9.4g}{dr:9.4g}{da / dr:6.2f}{cg:>11s}{cp:>14s}   "
              + " ".join(f"{u:.3g}" for u in g) + " / " + " ".join(f"{u:.3g}" for u in p))
        try:
            if 0.8 <= da / dr <= 1.25 and 0.4 <= float(cg) <= 1.2:
                good.append((k, da / dr, cg, cp))
        except ValueError:
            pass
    print("\n（目安: 音量の弁別閾はおおよそ 0.5〜1 dB、音程の弁別閾は定常音で数セント程度）")
    print("attack / release の比が 0.8〜1.25 かつ D* に等しい音量差が 0.4〜1.2 dB の距離:")
    for k, r, cg, cp in good:
        print(f"  {k:18s} 比 {r:.2f}  音量差 {cg} dB  音程差 {cp} セント")


if __name__ == "__main__":
    main()
