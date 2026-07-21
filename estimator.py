"""
estimator.py — 仕様 v7 §4/§5 準拠の推定器スケルトン
================================================================================
f_φ: (x, f0) → (ソースパラメータ, タイプ, ゲート, 深さ, θ_stat, θ_osc, G_mk)

構成:
  エンコーダ: log-mel スペクトログラム → 小型CNN → GAP → f0 条件を連結 → Dense
  7ヘッド（§4）:
    1. ENVパラメータ [K,5]（割合パラメータ化で a+h+d ≤ t_off を構造保証）
    2. LFOパラメータ [K,(rate,cosφ,sinφ,morph)] → スロット共用 [K,5] に整形
    3. ソースタイプ [K,2]（Gumbel-ST、§2.1b の混合と整合）
    4. ゲート [M,K+1]（Gumbel-ST、列スパース構造保証）
    5. 深さ [M]（tanh、D1）
    6. θ_stat [M]（fin 正準順序 (θ2,Δ) 構造保証）
    7. 静的のみ (θ_osc[2], G_mk[1])

損失（§5）: L = λ_spec·MRS + λ_param·||u_π*||²(λI版PNP) + λ_assign·(CE_gate+CE_type)
            + λ_sparse·Σ_m P[ẑ_m≠∅]
置換整合: K=4 なので全 4!=24 通りを総当たり（厳密ハンガリアン等価）。
          マッチングコスト = R行L2 + タイプ不一致（stop-gradient で π* を決定）。

注: M(θ) 事前計算後は param 項の二次形式を ū^T(M̄+λI)ū に差し替えるだけの構造。
"""
import numpy as np
import tensorflow as tf
from itertools import permutations

import diff_synth_tf as ds
from diff_synth_fast import synth_fast

K, M = ds.K_ENVS, ds.M_DEST
PERMS = tf.constant(list(permutations(range(K))), tf.int32)      # [24,K]
R_MAX = 0.6                                                       # release denorm 上限（生成と一致）
F0_LO = 65.4                                                      # MIDI36


def gumbel_st(logits, temp=1.0, hard=True):
    """straight-through Gumbel-Softmax（forward one-hot / backward soft）"""
    g = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 1e-9, 1.0)))
    y = tf.nn.softmax((logits + g) / temp, axis=-1)
    if hard:
        y_h = tf.one_hot(tf.argmax(y, axis=-1), tf.shape(logits)[-1], dtype=y.dtype)
        y = tf.stop_gradient(y_h - y) + y
    return y


class Estimator(tf.keras.Model):
    def __init__(self, sr=16000, t_off=1.0, arch="v2"):
        super().__init__()
        self.sr, self.t_off, self.arch = sr, t_off, arch
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(64, 513, sr, 30.0, 7800.0)
        if arch == "v2":
            # v8.2: 時間軸保存エンコーダ。周波数側を強くプーリング → BiGRU → attention pooling
            self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                          for c, st in ((32, (1, 2)), (64, (2, 2)),
                                        (96, (1, 2)), (128, (2, 2)))]
            self.bigru = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(128, return_sequences=True))
            self.att = tf.keras.layers.Dense(1)
        else:                                     # v1（最小骨格、比較用）
            self.convs = [tf.keras.layers.Conv2D(c, 3, 2, "same", activation="relu")
                          for c in (32, 64, 128)]
            self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.trunk = tf.keras.layers.Dense(256, activation="relu")
        d = tf.keras.layers.Dense
        self.h_env = d(K * 5)
        self.h_lfo = d(K * 4)
        self.h_type = d(K * 2)
        self.h_gate = d(M * (K + 1))
        self.h_depth = d(M)
        self.h_stat = d(M)
        self.h_static = d(3)

    def encode(self, x, f0):
        S = tf.abs(tf.signal.stft(x, 1024, 256, fft_length=1024))     # [B,T,513]
        mel = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)          # [B,T,64]
        h = mel[..., None]
        for c in self.convs:
            h = c(h)
        if self.arch == "v2":
            B = tf.shape(h)[0]
            h = tf.reshape(h, [B, -1, h.shape[2] * h.shape[3]])       # [B,T',F'·C]
            h = self.bigru(h)                                          # [B,T',256]
            w = tf.nn.softmax(self.att(h), axis=1)
            h = tf.concat([tf.reduce_sum(w * h, axis=1),
                           tf.reduce_mean(h, axis=1)], axis=-1)        # [B,512]
        else:
            h = self.gap(h)
        f0n = tf.math.log(f0 / F0_LO)[:, None] / np.log(16.0)         # MIDI36-84 → [0,1]
        return self.trunk(tf.concat([h, f0n], axis=-1))

    def call(self, x, f0, temp=1.0, hard=True):
        B = tf.shape(x)[0]
        h = self.encode(x, f0)
        # --- ENV（割合パラメータ化） ---
        ue = tf.sigmoid(tf.reshape(self.h_env(h), [B, K, 5]))
        a = tf.maximum(ue[..., 0] * self.t_off, 1e-3)
        hh = ue[..., 1] * (self.t_off - a)
        dd = tf.maximum(ue[..., 2] * (self.t_off - a - hh), 1e-3)
        s = ue[..., 3]
        r = ue[..., 4] * R_MAX + 1e-3
        p_env = tf.stack([a, hh, dd, s, r], axis=-1)                  # [B,K,5]
        # --- LFO（rate 対数 / 位相は (cos,sin) → 円環復元） ---
        ul = tf.reshape(self.h_lfo(h), [B, K, 4])
        rate = 0.1 * tf.pow(100.0, tf.sigmoid(ul[..., 0]))
        phase = tf.math.floormod(tf.atan2(ul[..., 2], ul[..., 1]) / (2 * np.pi), 1.0)
        morph = tf.sigmoid(ul[..., 3])
        zeros = tf.zeros_like(rate)
        p_lfo = tf.stack([rate, phase, morph, zeros, zeros], axis=-1)
        # --- タイプ（ST-Gumbel: forward one-hot → パラメータ選択が厳密） ---
        type_logits = tf.reshape(self.h_type(h), [B, K, 2])           # class0=ENV
        type_oh = gumbel_st(type_logits, temp, hard)
        w_env = type_oh[..., 0]                                       # [B,K]
        src_params = w_env[..., None] * p_env + (1 - w_env[..., None]) * p_lfo
        # --- ゲート・深さ ---
        gate_logits = tf.reshape(self.h_gate(h), [B, M, K + 1])
        gate_oh = gumbel_st(gate_logits, temp, hard)                  # [B,M,K+1]
        depth = tf.tanh(self.h_depth(h))                              # [B,M]
        R_hat = tf.transpose(gate_oh[..., :K], [0, 2, 1]) * depth[:, None, :]  # [B,K,M]
        # --- θ_stat（fin 正準順序を構造保証） ---
        us = tf.sigmoid(self.h_stat(h))                               # [B,M]
        f1 = us[:, 2] * 0.95                                          # 上限余白
        f2 = f1 + 0.05 + us[:, 3] * (1.0 - f1 - 0.05)
        theta = tf.concat([us[:, :2], f1[:, None], f2[:, None], us[:, 4:]], axis=1)
        # --- 静的のみ ---
        st = tf.sigmoid(self.h_static(h))
        return dict(src_params=src_params, type_logits=type_logits, type_oh=type_oh,
                    gate_logits=gate_logits, gate_oh=gate_oh, depth=depth,
                    R=R_hat, theta_stat=theta, theta_osc=st[:, :2], gmk=st[:, 2])


# ---------------------------------------------------------------- 置換整合（S4 総当たり）
def match_perm(pred, true_R, true_type):
    """π* = argmin_π Σ_k ||R̂_{π(k)}−R_k||² + w·type不一致。mapping[B,K] を返す"""
    Rp = tf.gather(tf.stop_gradient(pred["R"]), PERMS, axis=1)        # [B,24,K,M]
    cost = tf.reduce_sum((Rp - true_R[:, None]) ** 2, axis=[2, 3])
    tp = tf.gather(tf.stop_gradient(pred["type_oh"][..., 0]), PERMS, axis=1)  # [B,24,K]
    cost += 0.5 * tf.reduce_sum((tp - true_type[:, None]) ** 2, axis=2)
    idx = tf.argmin(cost, axis=1, output_type=tf.int32)               # [B]
    return tf.gather(PERMS, idx)                                      # [B,K]


def gather_rows(t, mapping):
    return tf.gather(t, mapping, axis=1, batch_dims=1)


# ---------------------------------------------------------------- 損失（§5）
SLOT_SCALE_ENV = tf.constant([1.0, 1.0, 1.0, 1.0, R_MAX])            # a,h,d[s], s, r


def losses(pred, batch, x_true, sr, t_off, n_harm, lam, matching=True):
    f0, true_R = batch["f0"], batch["R"]
    true_type = batch["src_type"]                                     # 1=ENV
    if matching:
        mapping = match_perm(pred, true_R, true_type)
    else:                                             # v8.2: 正準順序データでは恒等
        B = tf.shape(true_R)[0]
        mapping = tf.tile(tf.range(K, dtype=tf.int32)[None, :], [B, 1])

    src_p = gather_rows(pred["src_params"], mapping)                  # [B,K,5]
    type_lg = gather_rows(pred["type_logits"], mapping)
    R_hat = gather_rows(pred["R"], mapping)

    # --- λI 版 PNP（重みなし二乗和。M̄ 導入時に二次形式へ差し替え） ---
    l_stat = tf.reduce_mean(tf.reduce_sum((pred["theta_stat"] - batch["theta_stat"]) ** 2, -1))
    l_static = tf.reduce_mean(tf.reduce_sum((pred["theta_osc"] - batch["theta_osc"]) ** 2, -1)
                              + (pred["gmk"] - batch["gmk"]) ** 2)
    l_R = tf.reduce_mean(tf.reduce_sum((R_hat - true_R) ** 2, [1, 2]))
    is_env = true_type                                                # [B,K]
    d_env = tf.reduce_sum(((src_p - batch["src_params"]) / SLOT_SCALE_ENV) ** 2, -1)
    dphi = tf.abs(src_p[..., 1] - batch["src_params"][..., 1])
    circ = tf.minimum(dphi, 1.0 - dphi)
    lr_n = (tf.math.log(tf.maximum(src_p[..., 0], 1e-3) / 0.1)
            - tf.math.log(tf.maximum(batch["src_params"][..., 0], 1e-3) / 0.1)) / np.log(100.0)
    d_lfo = lr_n ** 2 + circ ** 2 + (src_p[..., 2] - batch["src_params"][..., 2]) ** 2
    l_src = tf.reduce_mean(tf.reduce_sum(is_env * d_env + (1 - is_env) * d_lfo, -1))
    l_param = l_stat + l_static + l_R + l_src

    # --- ゲート/型 CE（π* でクラス添字を揃える） ---
    gate_lg = pred["gate_logits"]                                     # [B,M,K+1]
    inv = tf.argsort(mapping, axis=1)                                 # true k → pred 行
    z = tf.cast(batch["z"], tf.int32)                                 # [B,M] (-1=∅)
    z_re = tf.where(z >= 0, tf.gather(inv, tf.maximum(z, 0), axis=1, batch_dims=1), K)
    l_gate = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=z_re, logits=gate_lg))
    t_lab = tf.cast(1.0 - true_type, tf.int32)                        # ENV→0, LFO→1
    l_type = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=t_lab, logits=type_lg))

    # --- 疎性（soft 確率で） ---
    p_assign = 1.0 - tf.nn.softmax(gate_lg, -1)[..., K]
    l_sparse = tf.reduce_mean(tf.reduce_sum(p_assign, -1))

    # --- スペクトル（再合成、ST one-hot で厳密分岐） ---
    x_hat = synth_fast(pred["theta_stat"], pred["theta_osc"], pred["gmk"],
                       pred["src_params"], pred["R"], f0, t_off,
                       int(x_true.shape[1]), sr, n_harm[0], n_harm[1],
                       src_type=pred["type_oh"][..., 0])
    l_spec = tf.reduce_mean(ds.mrstft_loss(x_hat, x_true))

    total = (lam["spec"] * l_spec + lam["param"] * l_param
             + lam["assign"] * (l_gate + l_type) + lam["sparse"] * l_sparse)
    return total, dict(spec=l_spec, param=l_param, gate=l_gate,
                       type=l_type, sparse=l_sparse), mapping
