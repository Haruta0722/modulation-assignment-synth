r"""env_only.py — ENV のパラメータだけを予測する小さなモデルをゼロから学習する（上限の確認）

  python env_only.py --mode pool --out env_only_pool     # B1: 今と同じエンコーダ（時間平均あり）
  python env_only.py --mode flat --out env_only_flat     # B2: 時間平均をせず全コマを次の層へ
  python env_only.py --mode tconv --out env_only_tconv   # 時間方向の畳み込みで縮める（mq / flat_reg も同様）

検証の二乗誤差が最小だった時点の重みで test を評価する（早期終了）。

使用中の ENV が 1 つだけの音に限り（この ENV は必ず音量の列につながる）、その ENV の
5 つのパラメータを割合の空間（attack/1 秒、hold/残り、decay/残り、sustain、release/0.6 秒）で学習する。
本体のモデルのような枠の割り当てやゲートはなく、ENV の損失だけで学習する。
"""
import argparse, json, os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

import estimator as E
import train as T

K = 4
NAMES = ["attack", "hold", "decay", "sustain", "release"]


def env_targets(data, t_off):
    """全サンプルについて (使用中の ENV が 1 つだけか, その ENV の割合 [5])"""
    P = data.P
    z, sp, st = P["z"].astype(int), P["src_params"], P["src_type"]
    used = np.stack([(z == k).any(1) for k in range(K)], 1)
    env1 = used & (st == 1)
    ok = env1.sum(1) == 1
    k = np.argmax(env1, 1)
    a, h, d, s, r = [sp[np.arange(len(k)), k, j] for j in range(5)]
    Y = np.stack([a / t_off, h / np.maximum(t_off - a, 1e-3), d / np.maximum(t_off - a - h, 1e-3),
                  s, r / E.R_MAX], 1).clip(0, 1).astype(np.float32)
    return ok, Y


class EnvNet(tf.keras.Model):
    """まとめ方（mode）
      pool     : 本体と同じ時間平均（注意の重み付き平均 ＋ 単純平均）
      flat     : 全コマを並べて全結合へ
      flat_reg : flat ＋ ドロップアウト
      tconv    : 時間方向の畳み込み（stride 2 を 2 回）で 23 → 6 コマに縮めてから並べる ＋ ドロップアウト
      mq       : 注目する時刻を学ぶクエリを n_query 個持ち、それぞれの重み付き和を並べる ＋ ドロップアウト
    """
    def __init__(self, sr, mode, drop=0.3, n_query=4, n_bins=0):
        super().__init__()
        self.mode, self.n_bins = mode, n_bins
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(64, 513, sr, 30.0, 7800.0)
        self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                      for c, st in ((32, (1, 2)), (64, (2, 2)), (96, (1, 2)), (128, (2, 2)))]
        self.bigru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))
        self.att = tf.keras.layers.Dense(1 if mode == "pool" else n_query)
        self.tconv = [tf.keras.layers.Conv1D(64, 3, 2, "same", activation="relu") for _ in range(2)]
        self.drop = tf.keras.layers.Dropout(0.0 if mode in ("pool", "flat") else drop)
        self.hid = tf.keras.layers.Dense(256 if mode in ("pool", "flat", "flat_reg") else 128,
                                         activation="relu")
        self.out_ = tf.keras.layers.Dense(5 * n_bins if n_bins else 5)

    def call(self, x, training=False):
        """n_bins = 0 なら 5 つの値（0〜1）、> 0 なら 5 × n_bins の区間のロジットを返す"""
        S = tf.abs(tf.signal.stft(x, 1024, 256, fft_length=1024))
        h = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)[..., None]
        for c in self.convs:
            h = c(h)
        B = tf.shape(h)[0]
        g = self.bigru(tf.reshape(h, [B, -1, h.shape[2] * h.shape[3]]))   # [B,T',256]
        if self.mode == "pool":
            w = tf.nn.softmax(self.att(g), axis=1)
            v = tf.concat([tf.reduce_sum(w * g, 1), tf.reduce_mean(g, 1)], -1)
        elif self.mode in ("flat", "flat_reg"):
            v = tf.reshape(g, [B, g.shape[1] * g.shape[2]])
        elif self.mode == "tconv":
            for c in self.tconv:
                g = c(g)
            v = tf.reshape(g, [B, g.shape[1] * g.shape[2]])
        else:                                                              # mq
            w = tf.nn.softmax(self.att(g), axis=1)                         # [B,T',Q] 時間方向に正規化
            v = tf.reshape(tf.einsum("btq,btc->bqc", w, g), [B, -1])       # [B,Q·256]
        v = self.drop(v, training=training)
        o = self.out_(self.drop(self.hid(v), training=training))
        if self.n_bins:
            return tf.reshape(o, [B, 5, self.n_bins])
        return tf.sigmoid(o)


def hl_gauss_target(y, n_bins, sigma_ratio=0.75):
    """HL-Gauss: 正解 y を中心とした正規分布を各区間に積分した確率 [B,5,n_bins]"""
    edges = tf.linspace(0.0, 1.0, n_bins + 1)
    sigma = sigma_ratio / n_bins
    cdf = 0.5 * (1 + tf.math.erf((edges[None, None, :] - y[..., None]) / (sigma * np.sqrt(2))))
    p = cdf[..., 1:] - cdf[..., :-1]
    return p / tf.reduce_sum(p, -1, keepdims=True)


def predict(model, data, idx, batch):
    """回帰: {'reg': [n,5]}、分布: {'mean','mode': [n,5], 'entropy': [n,5]（区間数で正規化、1 = 一様）}"""
    out = np.concatenate([model(data.batch(idx[s:s + batch])[0]).numpy()
                          for s in range(0, len(idx), batch)])
    if out.ndim == 2:
        return {"reg": out}
    nb = out.shape[-1]
    p = np.exp(out - out.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    centers = (np.arange(nb) + 0.5) / nb
    ent = -(p * np.log(np.maximum(p, 1e-12))).sum(-1) / np.log(nb)
    return {"mean": (p * centers).sum(-1), "mode": centers[p.argmax(-1)], "entropy": ent}


def scores(P, t):
    res = {}
    for j, n in enumerate(NAMES):
        err, err_c = np.abs(P[:, j] - t[:, j]), np.abs(np.median(t[:, j]) - t[:, j])
        res[n] = dict(corr=float(np.corrcoef(P[:, j], t[:, j])[0, 1]) if P[:, j].std() > 0 else 0.0,
                      ratio=float(P[:, j].std() / t[:, j].std()),
                      skill=float(1 - err.mean() / err_c.mean()))
    return res


def evaluate(model, data, idx, Y, batch, kind=None):
    pr = predict(model, data, idx, batch)
    return scores(pr[kind or next(iter(pr))], Y[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v7")
    ap.add_argument("--mode", choices=["pool", "flat", "flat_reg", "tconv", "mq"], required=True)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_bins", type=int, default=0,
                    help="> 0 なら回帰の代わりに区間の分類（HL-Gauss）で学習し、最頻値と平均の両方で評価")
    args = ap.parse_args()
    tf.random.set_seed(args.seed)

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    ok, Y = env_targets(data, T_OFF)
    tr = data.splits["train"][ok[data.splits["train"]]]
    va = data.splits["val"][ok[data.splits["val"]]][:2000]
    te = data.splits["test"][ok[data.splits["test"]]]
    print(f"使用中の ENV が 1 つだけの音: 学習 {len(tr)} / 検証 {len(va)} / test {len(te)}", flush=True)

    model = EnvNet(SR, args.mode, n_bins=args.n_bins)
    model(tf.zeros([1, N]))
    print(f"mode={args.mode} n_bins={args.n_bins}  全パラメータ {model.count_params():,}", flush=True)
    opt = tf.keras.optimizers.Adam(args.lr)

    def loss_fn(out, y):
        if args.n_bins:                                                    # 区間の交差エントロピー
            q = hl_gauss_target(y, args.n_bins)
            return tf.reduce_mean(tf.reduce_sum(-q * tf.nn.log_softmax(out, -1), [-2, -1]))
        return tf.reduce_mean(tf.reduce_sum((out - y) ** 2, -1))

    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(model(x, training=True), y)
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    def val_mse():
        return float(np.mean([loss_fn(model(data.batch(va[s:s + 128])[0]), tf.constant(Y[va[s:s + 128]]))
                              for s in range(0, len(va), 128)]))

    # 早期終了: 検証の損失（回帰は二乗誤差、分布は交差エントロピー）が最小の時点の重みを残す
    rng = np.random.default_rng(args.seed)
    best, best_it, best_w = np.inf, 0, None
    for it in range(args.steps + 1):
        idx = np.sort(rng.choice(tr, args.batch, replace=False))
        loss = step(data.batch(idx)[0], tf.constant(Y[idx]))
        if it % args.eval_every == 0:
            v = val_mse()
            if v < best:
                best, best_it, best_w = v, it, [w.numpy() for w in model.weights]
            r = evaluate(model, data, va, Y, 128)
            print(f"step {it:6d} loss={float(loss):.4f} val_mse={v:.4f}{' *' if best_it == it else '  '} [val 相関] "
                  + " ".join(f"{n}={r[n]['corr']:.3f}" for n in NAMES), flush=True)
    for w, b in zip(model.weights, best_w):
        w.assign(b)

    tr_sub = np.sort(rng.choice(tr, 2000, replace=False))
    pr_te, pr_tr = predict(model, data, te, 128), predict(model, data, tr_sub, 128)
    print(f"\n[test] mode={args.mode} n_bins={args.n_bins}（使用中の ENV が 1 つだけの音 {len(te)} 件、"
          f"検証が最良だった step {best_it} の重み）")
    out = dict(best_step=best_it)
    for kind in [k for k in ("reg", "mode", "mean") if k in pr_te]:
        r, r_tr = scores(pr_te[kind], Y[te]), scores(pr_tr[kind], Y[tr_sub])
        out[kind] = dict(test=r, train=r_tr)
        print(f"\n  予測 = {dict(reg='回帰の出力', mode='分布の最頻値', mean='分布の平均')[kind]}")
        print(f"{'パラメータ':10s}{'相関':>8s}{'ばらつき比':>12s}{'誤差の改善':>12s}{'学習データの相関':>18s}")
        for n in NAMES:
            print(f"{n:10s}{r[n]['corr']:8.3f}{r[n]['ratio']:12.3f}{r[n]['skill']:12.3f}{r_tr[n]['corr']:18.3f}")
    if "entropy" in pr_te:
        ent = pr_te["entropy"].mean(0)
        out["entropy"] = dict(zip(NAMES, ent.tolist()))
        print("\n  出力分布の広がり（正規化エントロピーの平均。1 = 一様 ＝ 何も読み取れていない）")
        print("  " + "  ".join(f"{n}={e:.3f}" for n, e in zip(NAMES, ent)))
    json.dump(out, open(args.out + ".json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
