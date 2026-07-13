"""
diff_synth_fast.py — 高速版微分可能シンセ（仕様 v5 準拠、diff_synth_tf と数値等価）
================================================================================
置き換えの要点:
  1) SVF / EQ biquad / コンプ追従は「線形時変再帰 s_n = A_n s_{n-1} + b_n」なので、
     Hillis–Steele 型の並列 affine scan で逐次深さ O(log N) に落とす（結合則:
     (A2,b2)∘(A1,b1) = (A2 A1, A2 b1 + b2)）。外部依存なし（tfp 不要）。
  2) コーラスはフィードバック遅延 D(n) ≥ D0（=20ms）を利用した厳密ブロック処理。
     チャンク長 C = floor(D0) 内の読み出しは必ず過去チャンクのみを参照するため、
     チャンク内は完全並列・結果は逐次版と厳密一致。逐次はチャンク数 T/20ms（75回）のみ。
     リングバッファ軌跡 O(N×L) の保存も消える。
  3) オシレータは saw/square で正弦波テーブルを共有（奇数次係数マスク）。

非線形な瞬時処理（tanh, softニー, denorm 等）は全て要素ごとで元から並列。
"""

import numpy as np
import tensorflow as tf

from diff_synth_tf import (K_ENVS, M_DEST, D0_MS, BETA_C, TAU_COMP, GAMMA_KNEE,
                           EPS_LOG, F_LOW, F_MID, F_HIGH, Q_MID, GMK_LO, GMK_HI,
                           dest_specs, smooth_clamp, adhsr, lfo, sources, modulate,
                           _rbj_coeffs, distortion, mrstft_loss)


# ---------------------------------------------------------------- 並列 affine scan
def affine_scan_scalar(A, b):
    """s_n = A_n s_{n-1} + b_n（s_0=0, inclusive）。A,b: [N,B] → s: [N,B]"""
    N = int(A.shape[0])
    offset = 1
    while offset < N:
        A_prev = tf.concat([tf.ones_like(A[:offset]), A[:-offset]], axis=0)
        b_prev = tf.concat([tf.zeros_like(b[:offset]), b[:-offset]], axis=0)
        b = A * b_prev + b
        A = A * A_prev
        offset *= 2
    return b


def affine_scan_2x2(A, b):
    """s_n = A_n s_{n-1} + b_n（s_0=0, inclusive）。A: [N,B,2,2], b: [N,B,2]"""
    N = int(A.shape[0])
    eye = tf.eye(2, batch_shape=[1, tf.shape(A)[1]])
    offset = 1
    while offset < N:
        A_prev = tf.concat([tf.tile(eye, [offset, 1, 1, 1]), A[:-offset]], axis=0)
        b_prev = tf.concat([tf.zeros_like(b[:offset]), b[:-offset]], axis=0)
        b = tf.linalg.matvec(A, b_prev) + b
        A = tf.matmul(A, A_prev)
        offset *= 2
    return b


def _shift_state(s):
    """inclusive scan の s_n から pre-update 状態 s_{n-1} を作る（先頭は 0）"""
    return tf.concat([tf.zeros_like(s[:1]), s[:-1]], axis=0)


# ---------------------------------------------------------------- 各段（高速版）
def svf_filter_fast(x, cutoff, drive_f, q, sr):
    """tanh drive → SVF(TPT)。状態 [ic1,ic2] の 2x2 affine scan。
    ic1' = (2a1−1)ic1 − 2a2 ic2 + 2a2 u
    ic2' = 2a2 ic1 + (1−2a3)ic2 + 2a3 u
    y    = a2 ic1_prev + (1−a3) ic2_prev + a3 u"""
    u = tf.tanh(drive_f * x)
    g = tf.tan(np.pi * cutoff / sr)
    k = 1.0 / q
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    uT, a1T, a2T, a3T = [tf.transpose(v) for v in (u, a1, a2, a3)]   # [N,B]
    A = tf.stack([tf.stack([2*a1T - 1.0, -2*a2T], -1),
                  tf.stack([2*a2T, 1.0 - 2*a3T], -1)], -2)            # [N,B,2,2]
    bv = tf.stack([2*a2T*uT, 2*a3T*uT], -1)                           # [N,B,2]
    s_prev = _shift_state(affine_scan_2x2(A, bv))
    y = a2T*s_prev[..., 0] + (1.0 - a3T)*s_prev[..., 1] + a3T*uT
    return tf.transpose(y)


def compressor_fast(x, thresh_db, ratio, g_mk_db, sr):
    """追従 ell はスカラー affine scan（係数 α は定数）、以降は要素ごと"""
    alpha = float(np.exp(-1.0 / (TAU_COMP * sr)))
    ell = affine_scan_scalar(alpha * tf.ones_like(tf.transpose(x)),
                             (1.0 - alpha) * tf.abs(tf.transpose(x)))
    ell = tf.transpose(ell)
    L_db = 20.0 * tf.math.log(ell + EPS_LOG) / np.log(10.0)
    over = tf.math.softplus(GAMMA_KNEE * (L_db - thresh_db)) / GAMMA_KNEE
    G_db = -(1.0 - 1.0 / ratio) * over
    return tf.pow(10.0, (G_db + g_mk_db[:, None]) / 20.0) * x


def _biquad_fast(x, coeffs):
    """TDF-II を affine scan で。z1' = −a1 z1 + z2 + (b1−a1 b0)x,
    z2' = −a2 z1 + (b2−a2 b0)x, y = b0 x + z1_prev"""
    b0, b1, b2, a1, a2 = [tf.transpose(c) for c in coeffs]            # [N,B]
    xT = tf.transpose(x)
    zeros, ones = tf.zeros_like(a1), tf.ones_like(a1)
    A = tf.stack([tf.stack([-a1, ones], -1),
                  tf.stack([-a2, zeros], -1)], -2)                    # [N,B,2,2]
    bv = tf.stack([(b1 - a1*b0)*xT, (b2 - a2*b0)*xT], -1)             # [N,B,2]
    z_prev = _shift_state(affine_scan_2x2(A, bv))
    return tf.transpose(b0*xT + z_prev[..., 0])


def eq3_fast(x, low_db, mid_db, high_db, sr):
    x = _biquad_fast(x, _rbj_coeffs("lowshelf", low_db, F_LOW, sr))
    x = _biquad_fast(x, _rbj_coeffs("peak", mid_db, F_MID, sr, q=Q_MID))
    x = _biquad_fast(x, _rbj_coeffs("highshelf", high_db, F_HIGH, sr))
    return x


def chorus_fast(x, rate, depth, feedback, sr):
    """厳密ブロック処理。D(n) ≥ D0 なのでチャンク C=floor(D0) 内の読み出しは
    必ず過去チャンクの b のみを参照 → チャンク内並列、結果は逐次版と一致。"""
    psi = tf.cumsum(rate, axis=1) / sr
    lfo = 0.5 * (1.0 + tf.sin(2.0 * np.pi * psi))
    D0 = D0_MS * 1e-3 * sr
    D = D0 + depth * lfo                                              # [B,N]
    N = int(x.shape[1])
    C = int(np.floor(D0))
    pad = int(np.ceil(D0 + 0.010 * sr)) + 4
    hist = tf.zeros([tf.shape(x)[0], pad])
    ys = []
    for t in range(int(np.ceil(N / C))):
        s0, s1 = t * C, min((t + 1) * C, N)
        xn, Dn, fbn = x[:, s0:s1], D[:, s0:s1], feedback[:, s0:s1]
        gpos = tf.range(s0, s1, dtype=tf.float32)[None, :]
        i = tf.floor(Dn)
        f = Dn - i
        cur_len = pad + s0
        pos1 = tf.clip_by_value(tf.cast(gpos - i, tf.int32) + pad, 0, cur_len - 1)
        pos2 = tf.clip_by_value(pos1 - 1, 0, cur_len - 1)
        w = (1.0 - f) * tf.gather(hist, pos1, batch_dims=1) \
            + f * tf.gather(hist, pos2, batch_dims=1)
        ys.append((1.0 - BETA_C) * xn + BETA_C * w)
        hist = tf.concat([hist, xn + fbn * w], axis=1)
    return tf.concat(ys, axis=1)


def oscillator_fast(p, theta_osc, f0, N, sr, n_harm_saw, n_harm_sqr):
    """saw/square で正弦波を共有（square は奇数次のみの係数マスク）"""
    out = 0.0
    levels = (p["level1"], p["level2"])
    fins = (p["fin1"], p["fin2"])
    hs = np.arange(1, n_harm_saw + 1, dtype=np.float32)
    saw_c = ((-1.0) ** (hs + 1)) / hs
    sqr_c = np.where((hs % 2 == 1) & (hs <= 2 * n_harm_sqr - 1), 1.0 / hs, 0.0).astype(np.float32)
    for i in range(2):
        f_inst = f0[:, None] * tf.pow(2.0, fins[i] / 1200.0)
        phi = tf.cumsum(f_inst, axis=1) / sr
        sins = tf.sin(2.0 * np.pi * phi[:, :, None] * hs[None, None, :])  # [B,N,H]
        saw = tf.reduce_sum(saw_c[None, None, :] * sins, axis=-1)
        sqr = tf.reduce_sum(sqr_c[None, None, :] * sins, axis=-1)
        a_i = theta_osc[:, i:i + 1]
        out = out + levels[i] * (a_i * saw + (1.0 - a_i) * sqr)
    return out


def synth_fast(theta_stat, theta_osc, g_mk_norm, env_params, R, f0, t_off,
               N, sr, n_harm_saw, n_harm_sqr, src_type=None):
    """順方向 g の高速版（diff_synth_tf.synth と同一シグネチャ・数値等価）"""
    env = sources(env_params, src_type, t_off, N, sr)
    p = modulate(theta_stat, R, env, sr)
    x = oscillator_fast(p, theta_osc, f0, N, sr, n_harm_saw, n_harm_sqr)
    x = svf_filter_fast(x, p["cutoff"], p["drive_f"], p["res_q"], sr)
    x = chorus_fast(x, p["rate"], p["depth"], p["feedback"], sr)
    g_mk_db = GMK_LO + g_mk_norm * (GMK_HI - GMK_LO)
    x = compressor_fast(x, p["thresh"], p["ratio"], g_mk_db, sr)
    x = eq3_fast(x, p["eq_low"], p["eq_mid"], p["eq_high"], sr)
    x = distortion(x, p["drive_d"], p["mix_d"])
    return x
