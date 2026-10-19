r"""rate_probe.py — 変調スペクトルだけから LFO のレートを学習する小さなモデル（実験 43）

本体（v25 / v27）はレートの相対誤差の中央値が 0.6 前後で、変調スペクトルの山を取るだけで当たる音でも外れる。
「情報はあるのに学習できない」のか、「この入力ではそもそも学べない」のかを、レートだけを学ぶ小さなモデルで分ける。

  MOD_LAYOUT=typed python rate_probe.py --make_feats                  # 特徴量を一度だけ計算して保存
  MOD_LAYOUT=typed python rate_probe.py --loss mse     --seed 0 --out probe43/mse_s0.json
  MOD_LAYOUT=typed python rate_probe.py --loss hlgauss --seed 0 --out probe43/hlgauss_s0.json
  MOD_LAYOUT=typed python rate_probe.py --summary probe43            # 一覧（本体 v25・学習なしの山取りと比べる）

対象: 使用中の LFO が 1 本だけの音（typed の枠 4）。どの LFO のレートかの曖昧さがない
入力: 変調スペクトル [82 段 × 16 組]（estimator.mod_spectrum、v27 と同じ）。学習用の平均・標準偏差で標準化
構造: 揺れの速さの方向の畳み込み（32、幅 5）× 2 → 全結合 256 → 出力（v27 の枝と同じ）
出力と損失（--loss）
  mse     : 正規化した対数レート y = log(r / 0.1) / log(100) ∈ [0, 1] を sigmoid で出し、2 乗誤差（本体と同じ）
  hlgauss : y を 128 区間に分けた分類。正解を幅 0.75 区間のガウスでぼかした分布との交差エントロピー（HL-Gauss）
            予測は確率が最大の区間の中心（最頻値）
評価: ±5% / ±10% 以内の割合、相対誤差の中央値、対数レートの相関（全体と 3〜10 Hz の帯の中）。
      レート帯ごと、変調先ごと（カットオフ・音量・ピッチを含むか）、簡単な音（カットオフを |深さ| 0.5 以上）
"""
import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import estimator as E

NB, SIG = 128, 0.75
FEAT = "rate_feats_{}.npz"


def norm_rate(r):
    return np.log(r / 0.1) / np.log(100.0)


def make_feats():
    from train import Data
    assert ds.LAYOUT == "typed", "MOD_LAYOUT=typed で実行する"
    D = Data("dataset_v7")
    mel_mat = tf.signal.linear_to_mel_weight_matrix(64, 513, D.man["sr"], 30.0, 7800.0)

    @tf.function
    def feat(x):
        S = tf.abs(tf.signal.stft(x, 1024, 256, fft_length=1024))
        return E.mod_spectrum(tf.math.log(tf.matmul(S, mel_mat) + 1e-5))

    for split in ("train", "val", "test"):
        idx = D.splits[split]
        z = D.P["z"][idx].astype(int)
        used = np.stack([(z == k).any(1) for k in range(ds.K_ENVS)], 1)
        sel = idx[used[:, ds.K_ENV_SLOTS:].sum(1) == 1]
        F = [feat(D.batch(sel[s0:s0 + 256])[0]).numpy() for s0 in range(0, len(sel), 256)]
        np.savez(FEAT.format(split), feats=np.concatenate(F).astype(np.float32), idx=sel,
                 rate=D.P["src_params"][sel, ds.K_ENV_SLOTS, 0], z=D.P["z"][sel],
                 R=D.P["R"][sel, ds.K_ENV_SLOTS])
        print(f"{split}: LFO 1 本の音 {len(sel)} 件 → {FEAT.format(split)}", flush=True)


class Probe(tf.keras.Model):
    def __init__(self, n_out):
        super().__init__()
        self.convs = [tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu") for _ in range(2)]
        self.d1 = tf.keras.layers.Dense(256, activation="relu")
        self.head = tf.keras.layers.Dense(n_out)

    def call(self, f):
        h = f
        for c in self.convs:
            h = c(h)
        return self.head(self.d1(tf.reshape(h, [tf.shape(h)[0], -1])))


def hl_target(y):
    """HL-Gauss: 正解 y ∈ [0,1] を幅 SIG 区間のガウスでぼかし、128 区間の確率にする"""
    edges = np.linspace(0.0, 1.0, NB + 1)
    s = SIG / NB
    c = 0.5 * (1 + tf.math.erf((edges[None, :] - y[:, None]) / (s * np.sqrt(2.0))).numpy())
    p = np.diff(c, axis=1)
    return (p / np.maximum(p.sum(1, keepdims=True), 1e-12)).astype(np.float32)


def predict(model, kind, F, bs=1024):
    out = np.concatenate([model(tf.constant(F[i:i + bs])).numpy() for i in range(0, len(F), bs)])
    y = 1 / (1 + np.exp(-out[:, 0])) if kind == "mse" else (np.argmax(out, 1) + 0.5) / NB
    return 0.1 * 100.0 ** y


def rate_stats(rp, rt):
    rel = np.abs(rp - rt) / rt
    fast = rt >= 3.0
    return dict(n=int(len(rt)), within5=float(np.mean(rel <= 0.05)), within10=float(np.mean(rel <= 0.1)),
                median_rel=float(np.median(rel)), corr=float(np.corrcoef(np.log(rp), np.log(rt))[0, 1]),
                corr_fast=float(np.corrcoef(np.log(rp[fast]), np.log(rt[fast]))[0, 1]) if fast.sum() > 2 else np.nan)


def groups(d):
    """評価の内訳: レート帯、変調先、簡単な音"""
    rt, z, R = d["rate"], d["z"].astype(int), d["R"]
    k = ds.K_ENV_SLOTS
    on = z == k
    g = {"全体": np.ones(len(rt), bool),
         "0.1〜1 Hz": rt < 1, "1〜3 Hz": (rt >= 1) & (rt < 3), "3〜10 Hz": rt >= 3,
         "カットオフを含む": on[:, 4], "音量を含む": on[:, 0] | on[:, 1], "ピッチを含む": on[:, 2] | on[:, 3],
         "簡単な音（カットオフ |深さ|≥0.5、1〜10 Hz）": on[:, 4] & (np.abs(R[:, 4]) >= 0.5) & (rt >= 1)}
    return g


def naive_peak(F):
    """学習なし: 帯域を合計した変調スペクトルの、0.9 Hz 以上で最大の山の位置"""
    freqs = np.arange(1, E.MOD_BINS + 1) * 62.5 / E.MOD_NFFT
    prof = F.sum(2)
    lo = freqs >= 0.9
    return freqs[lo][np.argmax(prof[:, lo], 1)]


def train(args):
    tf.random.set_seed(args.seed)                 # スレッド数は環境変数 TF_NUM_INTRAOP_THREADS で（estimator の読み込みで TF が初期化済み）
    rng = np.random.default_rng(args.seed)
    tr, va, te = (dict(np.load(FEAT.format(s))) for s in ("train", "val", "test"))
    mu, sd = tr["feats"].mean(0, keepdims=True), tr["feats"].std(0, keepdims=True) + 1e-6
    Ftr, Fva, Fte = ((d["feats"] - mu) / sd for d in (tr, va, te))
    ytr = norm_rate(tr["rate"]).astype(np.float32)
    ttr = hl_target(ytr) if args.loss == "hlgauss" else None
    model = Probe(1 if args.loss == "mse" else NB)
    model(tf.zeros([2, E.MOD_BINS, Ftr.shape[2]]))
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function
    def step(f, y, t):
        with tf.GradientTape() as tape:
            o = model(f)
            if args.loss == "mse":
                l = tf.reduce_mean((tf.sigmoid(o[:, 0]) - y) ** 2)
            else:
                l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=o))
        g = tape.gradient(l, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return l

    best, best_w, bad, hist = -1.0, None, 0, []
    n = len(ytr)
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        for s0 in range(0, n - args.batch + 1, args.batch):
            j = perm[s0:s0 + args.batch]
            l = step(tf.constant(Ftr[j]), tf.constant(ytr[j]),
                     tf.constant(ttr[j]) if ttr is not None else tf.zeros([len(j), 1]))
        v = rate_stats(predict(model, args.loss, Fva), va["rate"])
        hist.append(dict(epoch=ep, loss=float(l), **{f"val_{k}": x for k, x in v.items()}))
        print(f"  epoch {ep:3d} loss {float(l):.4f}  val ±10% 以内 {v['within10']:.3f}  相対誤差の中央値 "
              f"{v['median_rel']:.3f}", flush=True)
        if v["within10"] > best:
            best, best_w, bad = v["within10"], model.get_weights(), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.set_weights(best_w)
    rp = predict(model, args.loss, Fte)
    res = dict(args=vars(args), history=hist,
               test={name: rate_stats(rp[m], te["rate"][m]) for name, m in groups(te).items() if m.sum() > 2})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    t = res["test"]["全体"]
    print(f"test: ±5% {t['within5']:.3f} / ±10% {t['within10']:.3f} / 相対誤差の中央値 {t['median_rel']:.3f}")


def summary(root):
    te = dict(np.load(FEAT.format("test")))
    G = groups(te)
    rows = {}
    rows["学習なしの山取り"] = [{k: rate_stats(naive_peak(te["feats"])[m], te["rate"][m]) for k, m in G.items()}]
    # 本体 v25 / v27: 同じ音（test の LFO 1 本の音）での予測
    for tag in ("v25", "v27", "v28", "v29", "v30", "v31", "v32", "v33"):
        R_ = []
        for p in sorted(glob.glob(f"dump_{tag}_single_s*_test.npz")):
            d = np.load(p)
            pos = {i: j for j, i in enumerate(d["idx"])}
            j = np.array([pos[i] for i in te["idx"]])
            rp = d["sp_pred"][j, ds.K_ENV_SLOTS, 0]
            R_.append({k: rate_stats(rp[m], te["rate"][m]) for k, m in G.items()})
        if R_:
            rows[f"本体 {tag}"] = R_
    for kind in ("mse", "hlgauss"):
        fs = sorted(glob.glob(os.path.join(root, f"{kind}_s*.json")))
        if fs:
            rows[f"小さなモデル {kind}"] = [json.load(open(f))["test"] for f in fs]
    for gname in G:
        print(f"\n[{gname}]（{int(G[gname].sum())} 本）  ±5% 以内 / ±10% 以内 / 相対誤差の中央値 / 対数の相関 / 3〜10 Hz の帯の中の相関")
        for name, R_ in rows.items():
            vals = [r[gname] for r in R_ if gname in r]
            if not vals:
                continue
            m = {k: np.nanmean([v[k] for v in vals]) for k in ("within5", "within10", "median_rel", "corr", "corr_fast")}
            print(f"  {name:22s}{m['within5']:9.3f}{m['within10']:9.3f}{m['median_rel']:9.3f}{m['corr']:9.3f}"
                  f"{m['corr_fast']:9.3f}   （{len(vals)} 本の平均）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_feats", action="store_true")
    ap.add_argument("--summary", type=str, default="")
    ap.add_argument("--loss", choices=["mse", "hlgauss"], default="mse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    if args.make_feats:
        make_feats()
    elif args.summary:
        summary(args.summary)
    else:
        train(args)


if __name__ == "__main__":
    main()
