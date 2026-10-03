r"""probe_encoder.py — エンコーダのどの段階でソースのパラメータの情報が失われるかを調べる

  python probe_encoder.py --ckpt_dir ckpt_v11_single_s0

学習済みモデルを固定したまま途中の出力を取り出し、各段階の出力から線形回帰（リッジ回帰）で
パラメータを予測し、test での相関を比べる（プローブ）。
  (0) 入力の対数メル      [90 コマ × 64]
  (1) 畳み込みの後        [23 コマ × 128]（周波数方向は平均）
  (2) BiGRU の後          [23 コマ × 256]
  (3) 時間方向の平均の後  [512]（注意の重み付き平均 ＋ 単純平均）
  (4) trunk 層の後        [256]（各パラメータを予測する直前）
時間軸のある段階はコマを並べたまま 1 本のベクトルにして回帰する（時間の位置の情報が使える）。
順序の問題を避けるため、ENV は使用中の ENV が 1 つだけの音、LFO は使用中の LFO が 1 つだけの音に限る。
"""
import argparse, os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import estimator as E
import train as T
import weights_io as W

K = ds.K_ENVS
STAGES = ["(0) 入力の対数メル", "(1) 畳み込みの後", "(2) BiGRU の後", "(3) 時間平均の後", "(4) trunk の後"]
LAMBDAS = [1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5]


def encode_parts(model, x, f0):
    """Estimator.encode（v2）と同じ計算で、途中の出力を返す"""
    S = tf.abs(tf.signal.stft(x, 1024, 256, fft_length=1024))
    mel = tf.math.log(tf.matmul(S, model.mel_mat) + 1e-5)                 # [B,T,64]
    h = mel[..., None]
    for c in model.convs:
        h = c(h)
    conv = tf.reduce_mean(h, axis=2)                                      # [B,T',128]
    B = tf.shape(h)[0]
    g = model.bigru(tf.reshape(h, [B, -1, h.shape[2] * h.shape[3]]))      # [B,T',256]
    w = tf.nn.softmax(model.att(g), axis=1)
    pooled = tf.concat([tf.reduce_sum(w * g, 1), tf.reduce_mean(g, 1)], -1)
    f0n = tf.math.log(f0 / E.F0_LO)[:, None] / np.log(16.0)
    trunk = model.trunk(tf.concat([pooled, f0n], -1))
    flat = lambda t: tf.reshape(t, [B, -1]).numpy()
    return [flat(mel), flat(conv), flat(g), pooled.numpy(), trunk.numpy()]


def targets(data, idx, t_off):
    """1 つだけ使われている ENV / LFO の、割合空間のパラメータ（対象外は nan）"""
    P = data.P
    z, sp, st = P["z"][idx].astype(int), P["src_params"][idx], P["src_type"][idx]
    used = np.stack([(z == k).any(1) for k in range(K)], 1)
    env1, lfo1 = used & (st == 1), used & (st == 0)
    Y = np.full((len(idx), 7), np.nan, np.float32)
    for i in range(len(idx)):
        if env1[i].sum() == 1:
            a, h, d, s, r = sp[i, np.argmax(env1[i])]
            Y[i, :5] = (a / t_off, h / max(t_off - a, 1e-3), d / max(t_off - a - h, 1e-3), s, r / E.R_MAX)
        if lfo1[i].sum() == 1:
            rate, _, morph = sp[i, np.argmax(lfo1[i]), :3]
            Y[i, 5:] = (np.log(rate), morph)
    return Y


def model_pred(model, x, f0, idx, data):
    """モデル自身の出力（同じ番号の枠）を同じ割合空間で返す"""
    p = model(x, f0, temp=0.3, hard=True, noise=False)
    sp = p["src_params"].numpy()
    P = data.P
    z, st = P["z"][idx].astype(int), P["src_type"][idx]
    used = np.stack([(z == k).any(1) for k in range(K)], 1)
    env1, lfo1 = used & (st == 1), used & (st == 0)
    Y = np.full((len(idx), 7), np.nan, np.float32)
    for i in range(len(idx)):
        if env1[i].sum() == 1:
            a, h, d, s, r = sp[i, np.argmax(env1[i])]
            Y[i, :5] = (a, h / max(1 - a, 1e-3), d / max(1 - a - h, 1e-3), s, r / E.R_MAX)
        if lfo1[i].sum() == 1:
            rate, _, morph = sp[i, np.argmax(lfo1[i]), :3]
            Y[i, 5:] = (np.log(max(rate, 1e-6)), morph)
    return Y


def collect(model, data, idx, batch, t_off):
    feats, Ys, Ms = [[] for _ in STAGES], [], []
    for s0 in range(0, len(idx), batch):
        sel = idx[s0:s0 + batch]
        x, b = data.batch(sel)
        for j, f in enumerate(encode_parts(model, x, b["f0"])):
            feats[j].append(f.astype(np.float32))
        Ys.append(targets(data, sel, t_off))
        Ms.append(model_pred(model, x, b["f0"], sel, data))
    return [np.concatenate(f) for f in feats], np.concatenate(Ys), np.concatenate(Ms)


def ridge_fit_predict(Xtr, ytr, Xte, rng):
    """標準化 → 学習データの 2 割で λ を選ぶ → 全学習データで解き直して test を予測"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    n = len(Xtr)
    perm = rng.permutation(n)
    va, tr = perm[: n // 5], perm[n // 5:]

    def solve(X, y, lam):
        if X.shape[1] <= X.shape[0]:
            return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
        a = np.linalg.solve(X @ X.T + lam * np.eye(X.shape[0]), y)     # 次元 > 件数なら双対形
        return X.T @ a

    best = min(LAMBDAS, key=lambda lam: np.mean(
        (Xtr[va] @ solve(Xtr[tr].astype(np.float64), ytr[tr] - ym, lam) + ym - ytr[va]) ** 2))
    w = solve(Xtr.astype(np.float64), ytr - ym, best)
    return Xte @ w + ym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v7")
    ap.add_argument("--ckpt_dir", default="ckpt_v11_single_s0")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    model = E.Estimator(sr=SR, t_off=T_OFF, arch="v2")
    model(tf.zeros([1, N]), tf.constant([220.0]), temp=1.0, hard=True)
    W.load(model, W.latest(args.ckpt_dir))

    rng = np.random.default_rng(0)
    idx_tr = np.sort(rng.choice(data.splits["train"], args.n_train, replace=False))
    idx_te = data.splits["test"][:args.n_test]
    Ftr, Ytr, _ = collect(model, data, idx_tr, args.batch, T_OFF)
    Fte, Yte, Mte = collect(model, data, idx_te, args.batch, T_OFF)
    print("特徴の次元: " + ", ".join(f"{s.split()[0]} {f.shape[1]}" for s, f in zip(STAGES, Ftr)))

    names = ["ENV attack", "ENV hold", "ENV decay", "ENV sustain", "ENV release", "LFO rate(log)", "LFO morph"]
    print(f"\ntest での相関（学習 {args.n_train} 件・test {args.n_test} 件のうち、同じ型のソースが 1 つだけの音）")
    print(f"{'パラメータ':16s}{'件数(学習/test)':>16s}" + "".join(f"{s.split()[0]:>8s}" for s in STAGES) + f"{'モデル自身':>10s}")
    for j, name in enumerate(names):
        mtr, mte = ~np.isnan(Ytr[:, j]), ~np.isnan(Yte[:, j])
        row = f"{name:16s}{f'{mtr.sum()}/{mte.sum()}':>16s}"
        for F_tr, F_te in zip(Ftr, Fte):
            p = ridge_fit_predict(F_tr[mtr], Ytr[mtr, j], F_te[mte], rng)
            row += f"{np.corrcoef(p, Yte[mte, j])[0, 1]:8.3f}"
        mm = mte & ~np.isnan(Mte[:, j])
        row += f"{np.corrcoef(Mte[mm, j], Yte[mm, j])[0, 1]:10.3f}"
        print(row)
    print("\n" + "\n".join(STAGES))


if __name__ == "__main__":
    main()
