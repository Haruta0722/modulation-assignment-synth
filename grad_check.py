r"""grad_check.py — LFO のヘッドの損失が、どの層にどれだけの勾配として届くかの検算（実験 45・46）

  MOD_LAYOUT=typed python grad_check.py

ランダムな入力でモデルを作り、LFO の出力だけから作った損失の勾配を tf.GradientTape で計算して、
層ごとの重みの勾配の大きさ（L2 ノルム）を並べる。学習の良し悪しではなく、配線（勾配が止まる・流れる場所）の検算。

[実験 45] --lfo_grad α: trunk・BiGRU への勾配が 1 / 0.1 / 0 倍、LFO 専用の層・変調スペクトルの枝は不変、ENV のヘッドへは 0
[実験 46] --rate_ce_stop --rate_aux: 損失を 3 つに分けて、それぞれの勾配の届き先を見る
  レートの分類だけ    → 共有の層（trunk・BiGRU）へは 0、LFO 専用の層・変調スペクトルの枝へは届く
  位相・morph だけ    → 共有の層まで届く
  補助のレートの回帰だけ → 共有の層まで届く
"""
import numpy as np
import tensorflow as tf

import estimator as E

tf.random.set_seed(0)
x = tf.random.normal([4, 24000]) * 0.1
f0 = tf.constant([220.0] * 4)


def norms(m, loss_fn):
    groups = {"trunk": m.trunk, "BiGRU": m.bigru, "LFO 専用の層": m.lfo_trunk,
              "変調スペクトルの枝": m.mod_dense, "ENV のヘッド": m.h_env}
    with tf.GradientTape() as tape:
        out = m(x, f0, temp=1.0, hard=True, noise=False)
        loss = loss_fn(out)
    names = list(groups)
    vs = [v for n in names for v in groups[n].trainable_variables]
    gs = tape.gradient(loss, vs)
    i, txt = 0, []
    for n in names:
        k = len(groups[n].trainable_variables)
        norm = float(np.sqrt(sum(float(tf.reduce_sum(g ** 2)) for g in gs[i:i + k] if g is not None)))
        txt.append(f"{n} {norm:.3e}")
        i += k
    return "、".join(txt)


lfo = slice(4, 7)                       # typed の LFO の枠

print("[実験 45] --lfo_grad")
m = E.Estimator(arch="v4", lfo_read="mod", rate_head="hlgauss", lfo_grad=1.0)
m(x, f0, temp=1.0, hard=True, noise=False)
for a in (1.0, 0.1, 0.0):
    m.lfo_grad = a
    print(f"  lfo_grad={a}: " + norms(m, lambda o: tf.reduce_sum(o["rate_logits"] ** 2)
                                      + tf.reduce_sum(o["src_params"][:, lfo, 1:3] ** 2)))

print("[実験 46] --rate_ce_stop --rate_aux")
m = E.Estimator(arch="v4", lfo_read="mod", rate_head="hlgauss", rate_ce_stop=True, rate_aux=True)
m(x, f0, temp=1.0, hard=True, noise=False)
print("  レートの分類だけ      : " + norms(m, lambda o: tf.reduce_sum(o["rate_logits"][:, lfo] ** 2)))
print("  位相・morph だけ      : " + norms(m, lambda o: tf.reduce_sum(o["src_params"][:, lfo, 1:3] ** 2)))
print("  補助のレートの回帰だけ: " + norms(m, lambda o: tf.reduce_sum(tf.math.log(o["rate_reg"][:, lfo]) ** 2)))
