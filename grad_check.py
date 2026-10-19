r"""grad_check.py — --lfo_grad（実験 45）の検算: LFO だけの損失の勾配が、共有の層へ指定の倍率で届くか

  MOD_LAYOUT=typed python grad_check.py

lfo_grad = 1 / 0.1 / 0 で、trunk・BiGRU への勾配の大きさが 1 倍 / 1/10 / 0、
LFO 専用の層・変調スペクトルの枝への勾配は変わらない、ENV のヘッドへは 0 になるはず
"""
import numpy as np
import tensorflow as tf
import estimator as E

tf.random.set_seed(0)
m = E.Estimator(arch="v4", lfo_read="mod", rate_head="hlgauss", lfo_grad=1.0)
x = tf.random.normal([4, 24000]) * 0.1
f0 = tf.constant([220.0] * 4)
m(x, f0, temp=1.0, hard=True, noise=False)
groups = {"trunk": m.trunk.trainable_variables, "BiGRU": m.bigru.trainable_variables,
          "LFO 専用の層": m.lfo_trunk.trainable_variables, "変調スペクトルの枝": m.mod_dense.trainable_variables,
          "ENV のヘッド": m.h_env.trainable_variables}
for a in (1.0, 0.1, 0.0):
    m.lfo_grad = a
    with tf.GradientTape() as tape:
        out = m(x, f0, temp=1.0, hard=True, noise=False)
        loss = tf.reduce_sum(out["rate_logits"] ** 2) + tf.reduce_sum(out["src_params"][:, 4:, 1:3] ** 2)  # LFO だけの損失
    names = list(groups)
    gs = tape.gradient(loss, [v for n in names for v in groups[n]])
    i, txt = 0, []
    for n in names:
        k = len(groups[n])
        norm = float(np.sqrt(sum(float(tf.reduce_sum(g ** 2)) for g in gs[i:i + k] if g is not None)))
        txt.append(f"{n} {norm:.3e}")
        i += k
    print(f"lfo_grad={a}: " + "、".join(txt))
