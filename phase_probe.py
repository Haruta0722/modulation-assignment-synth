r"""phase_probe.py — 軌跡 → LFO パラメータ（レート・位相・morph）だけを学習する小さなモデルで、
Stage 2 の構造（時間方向のまとめ方）と位相の損失を比べる（実験 37）

  python phase_probe.py --arch pool --loss circ2 --train clean --out probe/pool_circ2_clean.json

構造（--arch）。いずれも Stage 2 と同じ BiGRU(64) の後に Dense(128) → 出力 4 つ（レート・位相 2 成分・morph）
  pool : BiGRU の出力を時間方向に平均・最大でまとめる（今の Stage 2 と同じ）
  ends : 順方向の最後の状態と、逆方向の最後の状態（= 時刻 0 の位置）を並べる
  flat : 時間方向にまとめず、全時刻を 1x1 畳み込みで 16 チャンネルに縮めて平らに並べる
位相の損失（--loss）。レートは対数の 2 乗誤差で共通
  circ2 : 円周上の距離の 2 乗 min(Δφ, 1−Δφ)²＋morph の 2 乗誤差（今の損失と同じ）
  cs    : (cos 2πφ, sin 2πφ) への 2 乗誤差 / 4π²（小さなずれで circ2 と同じ大きさ）＋morph の 2 乗誤差
  wave  : 予測の (レート, 位相, morph) から作った LFO 波形と、真の波形との平均絶対誤差
学習の入力（--train）
  clean : LFO の式から作るきれいな波形（レート・morph は学習用の音の真の値から、位相は一様に引き直す）
  synth : clean を実際の軌跡に似せて崩したもの。学習用の音の実際の軌跡を w ≈ α·e + β に当てはめ、
          別の音の (α, β, 残差) を新しい波形に適用する
  traj  : 学習用の音に Stage 1 を通した実際の軌跡
評価: 学習の入力と同じ種類の val で最良の重みを選び、test のきれいな波形と test の実際の軌跡で測る
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from lfo_phase_check import circ_stats, lin_stats, lfo_wave, RATE_BINS

SR, HOP, N = 16000, 256, 24000


def load_lfo(path):
    """dump から、使用中の真の LFO ソースの (真の波形, 軌跡, レート, 位相, morph)"""
    d = np.load(path)
    b, k = np.nonzero(d["used"] & (d["type_true"] == 0))
    return dict(e=d["e_true"][b, k].astype(np.float32), w=d["traj"][b, k].astype(np.float32),
                rate=d["sp_true"][b, k, 0].astype(np.float32),
                phase=d["sp_true"][b, k, 1].astype(np.float32),
                morph=d["sp_true"][b, k, 2].astype(np.float32))


def fit_degrade(D):
    """実際の軌跡を w ≈ α·e + β に当てはめ、(α, β, 残差) を返す"""
    e, w = D["e"], D["w"]
    ec = e - e.mean(1, keepdims=True)
    alpha = (ec * (w - w.mean(1, keepdims=True))).sum(1) / np.maximum((ec ** 2).sum(1), 1e-9)
    beta = w.mean(1) - alpha * e.mean(1)
    res = w - alpha[:, None] * e - beta[:, None]
    return alpha.astype(np.float32), beta.astype(np.float32), res.astype(np.float32)


def lfo_tf(rate, phase, morph, t):
    th = rate[:, None] * t[None, :] + phase[:, None]
    sin_w = tf.sin(2 * np.pi * th)
    tri_w = 1.0 - 4.0 * tf.abs(th + 0.25 - tf.floor(th + 0.75) - 0.5)
    m = morph[:, None]
    return 0.5 * (1.0 + m * sin_w + (1.0 - m) * tri_w)


class Probe(tf.keras.Model):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))
        if arch == "flat":
            self.red = tf.keras.layers.Conv1D(16, 1, activation="relu")
        self.pool = tf.keras.layers.Dense(128, activation="relu")
        self.head = tf.keras.layers.Dense(4)

    def call(self, w):
        h = self.gru(w[..., None])                                      # [B,T,128]
        if self.arch == "pool":
            g = tf.concat([tf.reduce_mean(h, 1), tf.reduce_max(h, 1)], -1)
        elif self.arch == "ends":
            g = tf.concat([h[:, -1, :64], h[:, 0, 64:]], -1)
        else:
            r = self.red(h)
            g = tf.reshape(r, [tf.shape(r)[0], r.shape[1] * r.shape[2]])
        ul = self.head(self.pool(g))
        rate = 0.1 * tf.pow(100.0, tf.sigmoid(ul[:, 0]))
        phase = tf.math.floormod(tf.atan2(ul[:, 2], ul[:, 1]) / (2 * np.pi), 1.0)
        return rate, phase, ul[:, 1:3], tf.sigmoid(ul[:, 3])


def make_loss(kind, t):
    def loss(pred, e, r, p, m):
        rate, phase, u, morph = pred
        lr = (tf.math.log(rate) - tf.math.log(r)) / np.log(100.0)
        l = tf.reduce_mean(lr ** 2)
        if kind == "circ2":
            dphi = tf.abs(phase - p)
            c = tf.minimum(dphi, 1.0 - dphi)
            l += tf.reduce_mean(c ** 2) + tf.reduce_mean((morph - m) ** 2)
        elif kind == "cs":
            tgt = tf.stack([tf.cos(2 * np.pi * p), tf.sin(2 * np.pi * p)], -1)
            l += tf.reduce_mean(tf.reduce_sum((u - tgt) ** 2, -1)) / (4 * np.pi ** 2) \
                + tf.reduce_mean((morph - m) ** 2)
        else:
            l += tf.reduce_mean(tf.abs(lfo_tf(rate, phase, morph, t) - e))
        return l
    return loss


def predict(model, x, bs=1024):
    out = [model(tf.constant(x[i:i + bs])) for i in range(0, len(x), bs)]
    return (np.concatenate([o[0].numpy() for o in out]), np.concatenate([o[1].numpy() for o in out]),
            np.concatenate([o[3].numpy() for o in out]))


def metrics(model, x, D):
    r, p, m = predict(model, x)
    out = {}
    for name, sel in [("all", np.ones(len(r), bool))] + \
            [(f"{lo}-{hi:.0f}", (D["rate"] >= lo) & (D["rate"] < hi)) for lo, hi in RATE_BINS]:
        c = circ_stats(p[sel], D["phase"][sel])
        lr = lin_stats(np.log(r[sel]), np.log(D["rate"][sel]))
        mo = lin_stats(m[sel], D["morph"][sel])
        out[name] = dict(phase_agree=c["agree"], phase_ratio=c["ratio"], phase_skill=c["skill"],
                         rate_corr=lr["agree"], rate_ratio=lr["ratio"], rate_skill=lr["skill"],
                         morph_corr=mo["agree"], morph_ratio=mo["ratio"], morph_skill=mo["skill"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["pool", "ends", "flat"], required=True)
    ap.add_argument("--loss", choices=["circ2", "cs", "wave"], required=True)
    ap.add_argument("--train", choices=["clean", "synth", "traj"], required=True)
    ap.add_argument("--train_dump", default="dump_v13_two_s0_train30k.npz")
    ap.add_argument("--val_dump", default="dump_v13_two_s0_val.npz")
    ap.add_argument("--test_dump", default="dump_v13_two_s0_test.npz")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    TR, VA, TE = load_lfo(args.train_dump), load_lfo(args.val_dump), load_lfo(args.test_dump)
    T = TR["e"].shape[1]
    t = (np.minimum(np.arange(T) * HOP + 512, N - 1) / SR).astype(np.float32)
    chk = np.abs(lfo_wave(TE["rate"], TE["phase"], TE["morph"], t) - TE["e"]).max()
    print(f"LFO 本数 train {len(TR['rate'])} / val {len(VA['rate'])} / test {len(TE['rate'])}、"
          f"式と真の波形の差の最大 {chk:.2e}", flush=True)
    alpha, beta, res = fit_degrade(TR)

    def sample(n):
        if args.train == "traj":
            j = rng.integers(0, len(TR["rate"]), n)
            return TR["w"][j], TR["e"][j], TR["rate"][j], TR["phase"][j], TR["morph"][j]
        j = rng.integers(0, len(TR["rate"]), n)
        r, m, p = TR["rate"][j], TR["morph"][j], rng.random(n).astype(np.float32)
        e = lfo_wave(r, p, m, t).astype(np.float32)
        if args.train == "clean":
            return e, e, r, p, m
        i = rng.integers(0, len(alpha), n)                       # 別の音の縮み・ずれ・残差
        return alpha[i, None] * e + beta[i, None] + res[i], e, r, p, m

    val_x = VA["e"] if args.train == "clean" else VA["w"]
    model = Probe(args.arch)
    model(tf.zeros([2, T]))
    opt = tf.keras.optimizers.Adam(args.lr)
    loss_fn = make_loss(args.loss, tf.constant(t))

    @tf.function
    def step(x, e, r, p, m):
        with tf.GradientTape() as tape:
            l = loss_fn(model(x), e, r, p, m)
        g = tape.gradient(l, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return l

    best, best_w, bad, hist, t0 = -np.inf, None, 0, [], time.time()
    for s in range(1, args.steps + 1):
        l = step(*[tf.constant(a) for a in sample(args.batch)])
        if s % args.eval_every == 0:
            _, p, _ = predict(model, val_x)
            a = circ_stats(p, VA["phase"])["agree"]
            hist.append(dict(step=s, loss=float(l), val_phase_agree=a))
            print(f"  step {s:6d}  loss {float(l):.4f}  val 位相の一致度 {a:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            if a > best:
                best, best_w, bad = a, model.get_weights(), 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
    model.set_weights(best_w)
    res_all = dict(args=vars(args), best_val_phase_agree=best, history=hist,
                   test_clean=metrics(model, TE["e"], TE), test_traj=metrics(model, TE["w"], TE),
                   val_clean=metrics(model, VA["e"], VA), val_traj=metrics(model, VA["w"], VA))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res_all, f, indent=1)
    for k in ("test_clean", "test_traj"):
        m = res_all[k]
        print(f"{k}: 位相の一致度 {m['all']['phase_agree']:.3f}（" +
              " / ".join(f"{b} Hz {m[b]['phase_agree']:.3f}" for b in list(m)[1:]) +
              f"）、ばらつき比 {m['all']['phase_ratio']:.2f}、レートの相関 {m['all']['rate_corr']:.3f}")


if __name__ == "__main__":
    main()
