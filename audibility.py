r"""audibility.py — ソースのパラメータを変えたとき、音（モデルの入力特徴）がどれだけ変わるか

  python audibility.py --data dataset_v7 --n 512

test の真のパラメータで音を作り直し（保存波形と一致することを確認）、1 か所だけ変えた音との距離を測る。
  attack±0.3s : 使用中の ENV 1 つのアタックを 0.3 秒ずらす（h, d は割合を保って作り直す）
  depth±0.1   : そのソースがつながる列の深さを 0 側へ 0.1 縮める（モデルが学習できている量の参照）
  depth±0.3   : 同じく 0.3
  rate×2      : 使用中の LFO 1 つのレートを 2 倍（5 Hz 超は 1/2）
距離はモデルの入力と同じ対数メル（64 帯域、窓 1024、ホップ 256）の平均絶対差と、MR-STFT 損失。
"""
import argparse, os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import train as T
from diff_synth_fast import synth_fast

K, M = ds.K_ENVS, ds.M_DEST
NAMES = [d[0] for d in ds.dest_specs(16000)]


def quantize(x, scale):
    """保存波形と同じ int16 量子化（モデルの入力と同じ条件で比べるため）"""
    return tf.round(tf.clip_by_value(x * scale, -32767, 32767)) / scale


def logmel(x, sr):
    S = tf.abs(tf.signal.stft(x, 1024, 256, fft_length=1024))
    W = tf.signal.linear_to_mel_weight_matrix(64, 513, sr, 30.0, 7800.0)
    return tf.math.log(tf.matmul(S, W) + 1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v7")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]

    def synth(b, sp=None, R=None):
        """生成時と同じく、音高ごとにナイキストを超えない倍音数で合成する"""
        sp = b["src_params"] if sp is None else sp
        R = b["R"] if R is None else R
        f0 = b["f0"].numpy()
        out = np.zeros((len(f0), N), np.float32)
        for f in np.unique(f0):
            s = np.nonzero(f0 == f)[0]
            fmax = float(f) * 2 ** (100 / 1200)
            hs, hq = int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2)
            g = lambda t: tf.gather(t, s)
            out[s] = synth_fast(g(b["theta_stat"]), g(b["theta_osc"]), g(b["gmk"]), g(sp), g(R),
                                g(b["f0"]), T_OFF, N, SR, hs, hq,
                                src_type=g(b["src_type"])).numpy()
        return tf.constant(out)

    rec = {k: [] for k in ("attack", "depth01", "depth03", "rate2")}
    meta = []                                                     # attack 変化の (列名の集合, |深さ|最大)
    sanity = []
    idx_all = data.splits["test"][:args.n]
    for s0 in range(0, len(idx_all), args.batch):
        idx = idx_all[s0:s0 + args.batch]
        x, b = data.batch(idx)
        x0 = quantize(synth(b), data.scale)
        sanity += list(10 * np.log10(tf.reduce_sum(x ** 2, 1).numpy()
                                     / np.maximum(tf.reduce_sum((x0 - x) ** 2, 1).numpy(), 1e-12)))
        L0 = logmel(x0, SR)
        sp, R = b["src_params"].numpy(), b["R"].numpy()
        styp, z = b["src_type"].numpy(), b["z"].numpy()
        Bn = len(idx)
        used = np.stack([(z == k).any(1) for k in range(K)], 1)

        def pick(cond):
            k = np.full(Bn, -1)
            for i in range(Bn):
                c = np.nonzero(cond[i])[0]
                if len(c):
                    k[i] = c[0]
            return k

        k_env = pick(used & (styp == 1))
        k_lfo = pick(used & (styp == 0))
        sp_a, sp_r = sp.copy(), sp.copy()
        R1, R3 = R.copy(), R.copy()
        for i in range(Bn):
            k = k_env[i]
            if k >= 0:
                a, h, d = sp[i, k, :3]
                u1 = h / max(T_OFF - a, 1e-6)
                u2 = d / max(T_OFF - a - h, 1e-6)
                a2 = a + 0.3 if a < 0.5 else a - 0.3
                h2 = u1 * (T_OFF - a2)
                d2 = max(u2 * (T_OFF - a2 - h2), 1e-3)
                sp_a[i, k, :3] = (a2, h2, d2)
                cols = np.nonzero(z[i] == k)[0]
                meta.append(({NAMES[c] for c in cols}, float(np.abs(R[i, k, cols]).max())))
                for Rx, delta in ((R1, 0.1), (R3, 0.3)):
                    for c in cols:
                        v = R[i, k, c]
                        Rx[i, k, c] = np.sign(v) * max(abs(v) - delta, 0.0)
            k = k_lfo[i]
            if k >= 0:
                r = sp[i, k, 0]
                sp_r[i, k, 0] = r * 2 if r <= 5 else r / 2
        for name, kw, sel in (("attack", dict(sp=tf.constant(sp_a)), k_env >= 0),
                              ("depth01", dict(R=tf.constant(R1)), k_env >= 0),
                              ("depth03", dict(R=tf.constant(R3)), k_env >= 0),
                              ("rate2", dict(sp=tf.constant(sp_r)), k_lfo >= 0)):
            x1 = quantize(synth(b, **kw), data.scale)
            dm = tf.reduce_mean(tf.abs(logmel(x1, SR) - L0), [1, 2]).numpy()
            dr = ds.mrstft_loss(x1, x0).numpy()
            rec[name] += list(zip(dm[sel], dr[sel]))
        print(f"  {s0 + Bn} / {len(idx_all)}", flush=True)

    print(f"\n再合成と保存波形の SNR（dB）最小 / 中央値: {min(sanity):.1f} / {np.median(sanity):.1f}")
    q = lambda v: " / ".join(f"{x:.3f}" for x in np.percentile(v, [25, 50, 75]))
    print("\n変化の種類       件数   対数メル距離(25/50/75%)        MR-STFT(25/50/75%)")
    for name, label in (("attack", "attack±0.3s"), ("depth01", "depth−0.1"),
                        ("depth03", "depth−0.3"), ("rate2", "rate×2")):
        v = np.array(rec[name])
        print(f"  {label:12s} {len(v):5d}   {q(v[:, 0]):26s}   {q(v[:, 1])}")

    v = np.array(rec["attack"])[:, 0]
    dref = np.array(rec["depth01"])[:, 0]
    print(f"\nattack±0.3s の対数メル距離が、同じ音の depth−0.1 より小さい割合: {100 * (v < dref).mean():.1f}%")
    print("\nattack±0.3s の対数メル距離（中央値）: ENV がつながる変調先別")
    by = {}
    for (cols, _), dv in zip(meta, v):
        for c in cols:
            by.setdefault(c, []).append(dv)
    for c in NAMES:
        if c in by and len(by[c]) >= 10:
            print(f"  {c:9s} n={len(by[c]):4d}  {np.median(by[c]):.3f}")


if __name__ == "__main__":
    main()
