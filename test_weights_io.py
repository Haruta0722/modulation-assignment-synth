"""weights_io の往復テスト: 保存 → 別初期化のモデルに読み込み → 全変数と出力が一致するか"""
import os, tempfile

import numpy as np
import tensorflow as tf

import estimator as E
import estimator2 as E2
import weights_io as W

SR, N, T_OFF = 16000, 16000, 1.0


def build(kind):
    m = E.Estimator(sr=SR, t_off=T_OFF, arch="v2") if kind == "single" \
        else E2.TwoStageEstimator(sr=SR, t_off=T_OFF, hop=256)
    m(tf.zeros([1, N]), tf.constant([220.0]), temp=1.0, hard=True)
    return m


def outputs(m, x, f0):
    p = m(x, f0, temp=0.3, hard=True, noise=False)
    return {k: v.numpy() for k, v in p.items()}


def main():
    x = tf.random.normal([2, N]) * 0.1
    f0 = tf.constant([110.0, 330.0])
    d = tempfile.mkdtemp()
    for kind in ("single", "two"):
        a, b = build(kind), build(kind)
        names = [getattr(v, "path", v.name) for v in W._model_vars(a)]
        differ = sum(not np.array_equal(u.numpy(), v.numpy())
                     for u, v in zip(W._model_vars(a), W._model_vars(b)))
        path = os.path.join(d, f"{kind}{W.SUFFIX}")
        W.save(a, path)
        W.load(b, path)
        for u, v, n in zip(W._model_vars(a), W._model_vars(b), names):
            assert np.array_equal(u.numpy(), v.numpy()), n
        oa, ob = outputs(a, x, f0), outputs(b, x, f0)
        for k in oa:
            assert np.array_equal(oa[k], ob[k]), k
        enc = [n for n in names if "conv" in n or "gru" in n]
        print(f"[{kind}] OK: {len(names)} 変数（うち conv/GRU {len(enc)}）、"
              f"読み込み前に異なっていた変数 {differ} 個 → 全て一致、出力も一致")


if __name__ == "__main__":
    main()
