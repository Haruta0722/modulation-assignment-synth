"""
estimator2.py — 二段構成の推定器（本タスク版）
================================================================================
診断（論文 4.2/4.3 節）で、パラメータを音から直接推定する構成が
性能を制約することが判明した:
    フレーム列を直接出力    shape_ok = 0.729
    ENV/LFO パラメータ経由  shape_ok = 0.333
    二段構成（診断系）      shape_ok = 0.359 → TTO で 0.417

本ファイルは、この二段構成を**本タスク**（17 変調先のゲート推定 ＋
4 ソースの同時推定）へ統合する。

設計:
  Stage 1: 音 → K 本の軌跡 e_k(n) をフレーム列として予測（[B,K,T]）
           教師 = 正準順序に並べた真の軌跡（信号空間 L1）
  Stage 2: 各軌跡 → ENV/LFO パラメータ ＋ 型
           **音を見ない**。K 本に同一の重みを適用（行ごとに独立）
  その他のヘッド（ゲート・深さ・θ_stat・静的値）は従来どおり音から推定

置換整合をどこで解くか:
  Stage 1 の出力を正準順序の軌跡として教師するため、
  Stage 2 は順序を意識しない。診断系での知見
  （Stage 2 は軌跡のみを入力とする）がそのまま活きる。

使い方:
  from estimator2 import TwoStageEstimator, losses2
  model = TwoStageEstimator(sr=SR, t_off=T_OFF)
"""
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast
from estimator import (K, M, PERMS, R_MAX, F0_LO, gumbel_st,
                       gather_rows, SLOT_SCALE_ENV, teacher_R)


class Stage1Traj(tf.keras.Model):
    """音 → K 本の軌跡（フレーム列）。時間軸を潰さない符号化器を用いる。"""

    def __init__(self, sr=16000, hop=256, mel=64, n_src=K):
        super().__init__()
        self.hop, self.n_src = hop, n_src
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(mel, 513, sr,
                                                             30.0, 7800.0)
        self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                      for c, st in ((32, (1, 2)), (64, (1, 2)),
                                    (96, (1, 2)), (128, (1, 2)))]
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=True))
        self.head = tf.keras.layers.Dense(n_src)      # K 本まとめて出力

    def call(self, x, f0):
        S = tf.abs(tf.signal.stft(x, 1024, self.hop, fft_length=1024))
        h = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)[..., None]
        for c in self.convs:
            h = c(h)
        B, T = tf.shape(h)[0], tf.shape(h)[1]
        h = tf.reshape(h, [B, T, h.shape[2] * h.shape[3]])
        f0n = tf.math.log(f0 / F0_LO)[:, None] / np.log(16.0)
        h = tf.concat([h, f0n[:, None, :] * tf.ones([1, T, 1])], -1)
        y = tf.sigmoid(self.head(self.bigru(h)))       # [B,T,K]
        return tf.transpose(y, [0, 2, 1])              # [B,K,T]


class Stage2Param(tf.keras.Model):
    """軌跡 → ENV/LFO パラメータ ＋ 型。音は見ない。K 本に同一重みを適用。"""

    def __init__(self, t_off=1.0):
        super().__init__()
        self.t_off = t_off
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(64, return_sequences=True))
        self.pool = tf.keras.layers.Dense(128, activation="relu")
        self.h_env = tf.keras.layers.Dense(5)
        self.h_lfo = tf.keras.layers.Dense(4)
        self.h_type = tf.keras.layers.Dense(2)

    def call(self, traj, temp=1.0, hard=True, noise=True):
        """traj: [B,K,T] → params [B,K,5], type_logits [B,K,2], type_oh [B,K,2]"""
        B, Ks, T = tf.shape(traj)[0], traj.shape[1], tf.shape(traj)[2]
        flat = tf.reshape(traj, [-1, T, 1])                  # [B*K,T,1]
        h = self.gru(flat)
        g = self.pool(tf.concat([tf.reduce_mean(h, 1), tf.reduce_max(h, 1)], -1))

        ue = tf.sigmoid(self.h_env(g))
        a = tf.maximum(ue[:, 0] * self.t_off, 1e-3)
        hh = ue[:, 1] * (self.t_off - a)
        dd = tf.maximum(ue[:, 2] * (self.t_off - a - hh), 1e-3)
        p_env = tf.stack([a, hh, dd, ue[:, 3], ue[:, 4] * R_MAX + 1e-3], -1)

        ul = self.h_lfo(g)
        rate = 0.1 * tf.pow(100.0, tf.sigmoid(ul[:, 0]))     # [0.1,10] Hz
        phase = tf.math.floormod(
            tf.atan2(ul[:, 2], ul[:, 1]) / (2 * np.pi), 1.0)
        z = tf.zeros_like(rate)
        p_lfo = tf.stack([rate, phase, tf.sigmoid(ul[:, 3]), z, z], -1)

        tl = self.h_type(g)                                   # [B*K,2] class0=ENV
        t_oh = gumbel_st(tl, temp, hard, noise)
        w = t_oh[:, 0:1]
        p = w[..., None] * p_env[:, None] + (1 - w[..., None]) * p_lfo[:, None]
        p = p[:, 0]                                           # [B*K,5]

        return (tf.reshape(p, [B, Ks, 5]),
                tf.reshape(tl, [B, Ks, 2]),
                tf.reshape(t_oh, [B, Ks, 2]))


class TwoStageEstimator(tf.keras.Model):
    """本タスク用の二段構成推定器。
    Stage 1/2 が ENV/LFO パラメータと型を担い、
    ゲート・深さ・静的値は音から直接推定する（従来と同一）。"""

    def __init__(self, sr=16000, t_off=1.0, hop=256):
        super().__init__()
        self.sr, self.t_off, self.hop = sr, t_off, hop
        self.stage1 = Stage1Traj(sr, hop)
        self.stage2 = Stage2Param(t_off)
        # ゲート・深さ・静的値用の符号化器（時間方向を要約）
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(64, 513, sr,
                                                             30.0, 7800.0)
        self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                      for c, st in ((32, (1, 2)), (64, (2, 2)),
                                    (96, (1, 2)), (128, (2, 2)))]
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=True))
        self.att = tf.keras.layers.Dense(1)
        self.trunk = tf.keras.layers.Dense(256, activation="relu")
        d = tf.keras.layers.Dense
        self.h_gate = d(M * (K + 1))
        self.h_depth = d(M)
        self.h_stat = d(M)
        self.h_static = d(3)

    def encode(self, x, f0):
        S = tf.abs(tf.signal.stft(x, 1024, 256, fft_length=1024))
        h = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)[..., None]
        for c in self.convs:
            h = c(h)
        B = tf.shape(h)[0]
        h = tf.reshape(h, [B, -1, h.shape[2] * h.shape[3]])
        h = self.bigru(h)
        w = tf.nn.softmax(self.att(h), axis=1)
        h = tf.concat([tf.reduce_sum(w * h, 1), tf.reduce_mean(h, 1)], -1)
        f0n = tf.math.log(f0 / F0_LO)[:, None] / np.log(16.0)
        return self.trunk(tf.concat([h, f0n], -1))

    def call(self, x, f0, temp=1.0, hard=True, noise=True):
        B = tf.shape(x)[0]
        # --- 二段: 軌跡 → パラメータ ---
        traj = self.stage1(x, f0)                             # [B,K,T]
        src_params, type_logits, type_oh = self.stage2(
            traj, temp=temp, hard=hard, noise=noise)
        # --- 従来どおりのヘッド ---
        h = self.encode(x, f0)
        gate_logits = tf.reshape(self.h_gate(h), [B, M, K + 1])
        gate_oh = gumbel_st(gate_logits, temp, hard, noise)
        depth = tf.tanh(self.h_depth(h))
        R_hat = tf.transpose(gate_oh[..., :K], [0, 2, 1]) * depth[:, None, :]
        us = tf.sigmoid(self.h_stat(h))
        f1 = us[:, 2] * 0.95
        f2 = f1 + 0.05 + us[:, 3] * (1.0 - f1 - 0.05)
        theta = tf.concat([us[:, :2], f1[:, None], f2[:, None], us[:, 4:]], 1)
        st = tf.sigmoid(self.h_static(h))
        return dict(traj=traj, src_params=src_params,
                    type_logits=type_logits, type_oh=type_oh,
                    gate_logits=gate_logits, gate_oh=gate_oh, depth=depth,
                    R=R_hat, theta_stat=theta,
                    theta_osc=st[:, :2], gmk=st[:, 2])


def losses2(pred, batch, x_true, sr, t_off, n_harm, lam, hop=256):
    """本タスク用の損失。estimator.losses に Stage 1 の軌跡教師を加えた版。
    正準順序データを前提とするため置換整合は恒等（matching=False 相当）。"""
    N = int(x_true.shape[1])
    B = tf.shape(x_true)[0]
    f0, true_R = batch["f0"], batch["R"]
    true_type = batch["src_type"]                             # 1=ENV

    # ---- Stage 1: 軌跡の教師（信号空間、使用中ソースのみ） ----
    e_true = ds.sources(batch["src_params"], batch["src_type"],
                        t_off, N, sr)                         # [B,K,N]
    idx = tf.minimum(tf.range(tf.shape(pred["traj"])[2]) * hop + 512, N - 1)
    y_traj = tf.gather(e_true, idx, axis=2)                   # [B,K,T]
    z_all = tf.cast(batch["z"], tf.int32)
    used = tf.cast(tf.reduce_any(
        tf.equal(z_all[:, :, None], tf.range(K, dtype=tf.int32)[None, None, :]),
        axis=1), tf.float32)                                  # [B,K]
    nu = tf.maximum(tf.reduce_sum(used), 1.0)
    l_traj = tf.reduce_sum(
        used * tf.reduce_mean(tf.abs(pred["traj"] - y_traj), -1)) / nu

    # ---- パラメータ（従来と同じ、used マスクつき） ----
    sp, tl = pred["src_params"], pred["type_logits"]
    d_env = tf.reduce_sum(
        (((sp - batch["src_params"]) / SLOT_SCALE_ENV) ** 2), -1)
    dphi = tf.abs(sp[..., 1] - batch["src_params"][..., 1])
    circ = tf.minimum(dphi, 1.0 - dphi)
    lr_n = (tf.math.log(tf.maximum(sp[..., 0], 1e-3) / 0.1)
            - tf.math.log(tf.maximum(batch["src_params"][..., 0], 1e-3) / 0.1)
            ) / np.log(100.0)
    d_lfo = lr_n ** 2 + circ ** 2 + (sp[..., 2] - batch["src_params"][..., 2]) ** 2
    l_src = tf.reduce_sum(
        used * (true_type * d_env + (1 - true_type) * d_lfo)) / nu

    l_stat = tf.reduce_mean(tf.reduce_sum(
        (pred["theta_stat"] - batch["theta_stat"]) ** 2, -1))
    l_static = tf.reduce_mean(tf.reduce_sum(
        (pred["theta_osc"] - batch["theta_osc"]) ** 2, -1)
        + (pred["gmk"] - batch["gmk"]) ** 2)
    R_hat = (teacher_R(pred["depth"], batch["z"]) if lam.get("depth_tf", 0)
             else pred["R"])
    l_R = tf.reduce_mean(tf.reduce_sum((R_hat - true_R) ** 2, [1, 2]))
    l_param = l_stat + l_static + l_R + l_src

    # ---- ゲート・型 CE（正準順序のため恒等対応） ----
    gate_lg = pred["gate_logits"]
    z_re = tf.where(z_all >= 0, z_all, K)
    ce_g = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=z_re, logits=gate_lg)
    w_none = lam.get("none_w", 1.0)
    wg = tf.where(tf.equal(z_re, K), w_none * tf.ones_like(ce_g),
                  tf.ones_like(ce_g))
    l_gate = tf.reduce_sum(wg * ce_g) / tf.maximum(tf.reduce_sum(wg), 1e-6)

    t_lab = tf.cast(1.0 - true_type, tf.int32)
    ce_t = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=t_lab, logits=tl)
    l_type = tf.reduce_sum(used * ce_t) / nu

    p_assign = 1.0 - tf.nn.softmax(gate_lg, -1)[..., K]
    l_sparse = tf.reduce_mean(tf.reduce_sum(p_assign, -1))

    # ---- スペクトル（既定では無効） ----
    if lam["spec"] <= 0.0:
        l_spec = tf.constant(0.0)
    else:
        x_hat = synth_fast(pred["theta_stat"], pred["theta_osc"], pred["gmk"],
                           pred["src_params"], pred["R"], f0, t_off, N, sr,
                           n_harm[0], n_harm[1],
                           src_type=pred["type_oh"][..., 0])
        l_spec = tf.reduce_mean(ds.mrstft_loss(x_hat, x_true))

    total = (lam["spec"] * l_spec + lam["param"] * l_param
             + lam["assign"] * (l_gate + l_type) + lam["sparse"] * l_sparse
             + lam.get("traj", 1.0) * l_traj)
    mapping = tf.tile(tf.range(K, dtype=tf.int32)[None, :], [B, 1])
    return total, dict(spec=l_spec, param=l_param, gate=l_gate,
                       type=l_type, sparse=l_sparse, traj=l_traj), mapping
