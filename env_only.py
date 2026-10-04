r"""env_only.py — ENV のパラメータだけを予測する小さなモデルをゼロから学習する（上限の確認）

  python env_only.py --mode pool --out env_only_pool     # B1: 今と同じエンコーダ（時間平均あり）
  python env_only.py --mode flat --out env_only_flat     # B2: 時間平均をせず全コマを次の層へ

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
    def __init__(self, sr, mode):
        super().__init__()
        self.mode = mode
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(64, 513, sr, 30.0, 7800.0)
        self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                      for c, st in ((32, (1, 2)), (64, (2, 2)), (96, (1, 2)), (128, (2, 2)))]
        self.bigru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))
        self.att = tf.keras.layers.Dense(1)
        self.hid = tf.keras.layers.Dense(256, activation="relu")
        self.out_ = tf.keras.layers.Dense(5)

    def call(self, x):
        S = tf.abs(tf.signal.stft(x, 1024, 256, fft_length=1024))
        h = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)[..., None]
        for c in self.convs:
            h = c(h)
        B = tf.shape(h)[0]
        g = self.bigru(tf.reshape(h, [B, -1, h.shape[2] * h.shape[3]]))   # [B,T',256]
        if self.mode == "pool":                                            # 本体と同じ時間平均
            w = tf.nn.softmax(self.att(g), axis=1)
            v = tf.concat([tf.reduce_sum(w * g, 1), tf.reduce_mean(g, 1)], -1)
        else:                                                              # 全コマを並べたまま
            v = tf.reshape(g, [B, g.shape[1] * g.shape[2]])
        return tf.sigmoid(self.out_(self.hid(v)))


def evaluate(model, data, idx, Y, batch):
    P = np.concatenate([model(data.batch(idx[s:s + batch])[0]).numpy()
                        for s in range(0, len(idx), batch)])
    t = Y[idx]
    res = {}
    for j, n in enumerate(NAMES):
        err, err_c = np.abs(P[:, j] - t[:, j]), np.abs(np.median(t[:, j]) - t[:, j])
        res[n] = dict(corr=float(np.corrcoef(P[:, j], t[:, j])[0, 1]),
                      ratio=float(P[:, j].std() / t[:, j].std()),
                      skill=float(1 - err.mean() / err_c.mean()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v7")
    ap.add_argument("--mode", choices=["pool", "flat"], required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tf.random.set_seed(args.seed)

    data = T.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    ok, Y = env_targets(data, T_OFF)
    tr = data.splits["train"][ok[data.splits["train"]]]
    va = data.splits["val"][ok[data.splits["val"]]][:1000]
    te = data.splits["test"][ok[data.splits["test"]]]
    print(f"使用中の ENV が 1 つだけの音: 学習 {len(tr)} / 検証 {len(va)} / test {len(te)}", flush=True)

    model = EnvNet(SR, args.mode)
    model(tf.zeros([1, N]))
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.reduce_sum((model(x) - y) ** 2, -1))
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    rng = np.random.default_rng(args.seed)
    for it in range(args.steps + 1):
        idx = np.sort(rng.choice(tr, args.batch, replace=False))
        loss = step(data.batch(idx)[0], tf.constant(Y[idx]))
        if it % 3000 == 0:
            r = evaluate(model, data, va, Y, 128)
            print(f"step {it:6d} loss={float(loss):.4f}  [val 相関] "
                  + " ".join(f"{n}={r[n]['corr']:.3f}" for n in NAMES), flush=True)

    r = evaluate(model, data, te, Y, 128)
    print(f"\n[test] mode={args.mode}（使用中の ENV が 1 つだけの音 {len(te)} 件）")
    print(f"{'パラメータ':10s}{'相関':>8s}{'ばらつき比':>12s}{'誤差の改善':>12s}")
    for n in NAMES:
        print(f"{n:10s}{r[n]['corr']:8.3f}{r[n]['ratio']:12.3f}{r[n]['skill']:12.3f}")
    json.dump(r, open(args.out + ".json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
