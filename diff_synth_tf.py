"""
diff_synth_tf.py — フェーズ1 定式化仕様書 v5 に基づく微分可能シンセ（TensorFlow）
================================================================================
順方向モデル g(θ_stat, θ_osc, G_mk, E, R) → x ∈ R^[B,N]

仕様対応:
  §2.1 ADHSR（5パラメータ、区分線形、a+h+d ≤ t_off は呼び出し側で担保）
  §2.2 smooth clamp σ（softplus 型、β=10）、正規化制御空間 → denorm
  §2.3 saw↔square モーフ2オシレータ（位相累積・帯域制限・正準デチューン順序は生成側で担保）
  §2.4 SVF(TPT)+drive / chorus(変調ディレイ+FB) / compressor(softニー+G_mk) /
       3バンドEQ(RBJ biquad, TDF-II) / distortion(tanh)
  denorm レンジは §3 対応表（cutoff 上限は 0.45·sr に修正済み）

実装ノート:
  - 再帰段（SVF/chorus/comp/EQ）は tf.scan。BPTT の活性化保持がメモリ律速（§9.4）。
  - R は連続 [B,K,M] 行列として受ける（列スパースは呼び出し側の表現。§5.2 と整合）。
  - すべて float32。
"""

import os

import numpy as np
import tensorflow as tf

# ソースの枠の並び（実験 41）。環境変数 MOD_LAYOUT で切り替える
#   mixed: 型（ENV / LFO）が混在する 4 枠。正準順序は「使用中の ENV を attack 順 → 使用中の LFO をレート順」（従来）
#   typed: ENV 4 枠 ＋ LFO 3 枠に型を固定。LFO の枠の位置が ENV の本数に左右されない（実シンセの構成に近い）
#          データセット（4 ソース、型は混在）は読み込み時に並べ替える（train.Data）
LAYOUT = os.environ.get("MOD_LAYOUT", "mixed")
if LAYOUT not in ("mixed", "typed"):
    raise ValueError(f"MOD_LAYOUT={LAYOUT}（mixed か typed）")
K_ENV_SLOTS, K_LFO_SLOTS = 4, 3
K_ENVS = 4 if LAYOUT == "mixed" else K_ENV_SLOTS + K_LFO_SLOTS
SLOT_TYPE = (None if LAYOUT == "mixed"                                   # 枠ごとの型（1=ENV, 0=LFO）
             else np.array([1.0] * K_ENV_SLOTS + [0.0] * K_LFO_SLOTS, np.float32))
M_DEST = 17
BETA_CLAMP = 10.0

# ----- §3 変調先対応表 (name, kind, lo, hi) -- kind: 'lin' or 'log' -----
def dest_specs(sr):
    return [
        ("level1",   "lin",  0.0,    1.0),
        ("level2",   "lin",  0.0,    1.0),
        ("fin1",     "lin", -100.0, 100.0),   # cents
        ("fin2",     "lin", -100.0, 100.0),
        ("cutoff",   "log", 20.0,   0.45 * sr),
        ("drive_f",  "log",  1.0,   10.0),   # v6: 対数化（tanh 飽和対策、スパイク実測に基づく）
        ("res_q",    "lin",  0.5,   10.0),
        ("rate",     "log",  0.1,   10.0),    # Hz
        ("depth",    "lin",  0.0,   0.010 * sr),  # samples (10 ms)
        ("feedback", "lin",  0.0,    0.9),
        ("thresh",   "lin", -60.0,   0.0),    # dB
        ("ratio",    "lin",  1.0,   20.0),
        ("eq_low",   "lin", -12.0,  12.0),    # dB
        ("eq_mid",   "lin", -12.0,  12.0),
        ("eq_high",  "lin", -12.0,  12.0),
        ("drive_d",  "log",  1.0,   10.0),   # v6: 対数化（同上）
        ("mix_d",    "lin",  0.0,    1.0),
    ]

# §2.4 固定定数
D0_MS = 20.0          # chorus 基本ディレイ
BETA_C = 0.5          # chorus wet 比
TAU_COMP = 0.010      # comp 追従時定数 [s]
GAMMA_KNEE = 0.5      # softニー鋭さ
EPS_LOG = 1e-6
F_LOW, F_MID, F_HIGH = 200.0, 1000.0, 5000.0
Q_MID = 0.7
GMK_LO, GMK_HI = 0.0, 12.0   # makeup [dB]


def smooth_clamp(u, beta=BETA_CLAMP):
    """σ_[0,1](u) = (1/β)(softplus(βu) − softplus(β(u−1)))  — §2.2"""
    return (tf.math.softplus(beta * u) - tf.math.softplus(beta * (u - 1.0))) / beta


def adhsr(env_params, t_off, N, sr):
    """§2.1 区分線形 ADHSR。env_params: [B,K,5]=(a,h,d,s,r) 秒/レベル → e:[B,K,N]"""
    a = tf.maximum(env_params[..., 0:1], 1e-4)
    h = tf.maximum(env_params[..., 1:2], 0.0)
    d = tf.maximum(env_params[..., 2:3], 1e-4)
    s = env_params[..., 3:4]
    r = tf.maximum(env_params[..., 4:5], 1e-4)
    t = tf.range(N, dtype=tf.float32)[None, None, :] / sr        # [1,1,N]
    e_att = t / a
    e_dec = 1.0 - (1.0 - s) * (t - a - h) / d
    e_rel = s * (1.0 - (t - t_off) / r)
    e = tf.where(t < a, e_att,
        tf.where(t < a + h, tf.ones_like(t) * tf.ones_like(a),
        tf.where(t < a + h + d, e_dec,
        tf.where(t < t_off, s * tf.ones_like(t),
        tf.where(t < t_off + r, e_rel, tf.zeros_like(t))))))
    return tf.clip_by_value(e, 0.0, 1.0)


def lfo(src_params, N, sr):
    """§2.1b 周期LFO。スロット共用: [...,0]=rate[Hz], [...,1]=phase[0,1), [...,2]=morph m
    ℓ = ½(1 + m·sin(2πθ) + (1−m)·tri(θ)),  θ = f·t + φ,  tri は sin と位相整列した三角波。
    振幅は構造的に [0,1]（振幅ゲージなし）。"""
    f = src_params[..., 0:1]
    ph = src_params[..., 1:2]
    m = src_params[..., 2:3]
    t = tf.range(N, dtype=tf.float32)[None, None, :] / sr
    th = f * t + ph
    sin_w = tf.sin(2.0 * np.pi * th)
    tri_w = 1.0 - 4.0 * tf.abs(th + 0.25 - tf.floor(th + 0.75) - 0.5)
    return 0.5 * (1.0 + m * sin_w + (1.0 - m) * tri_w)


def sources(src_params, src_type, t_off, N, sr):
    """変調ソース e_k(n) を生成（v7）。src_type: [B,K] 1.0=ENV(ADHSR) / 0.0=LFO。
    None なら全ソース ENV（v6 互換）。学習時は Gumbel 緩和の連続値でも可。"""
    e_env = adhsr(src_params, t_off, N, sr)
    if src_type is None:
        return e_env
    w = src_type[:, :, None]
    return w * e_env + (1.0 - w) * lfo(src_params, N, sr)


def modulate(theta_stat, R, env, sr):
    """§2.2 c_m = σ(θ_m + Σ_k R_km e_k) → denorm。
    theta_stat:[B,M] R:[B,K,M] env:[B,K,N] → dict name→[B,N] 物理値"""
    mod = tf.einsum('bkm,bkn->bmn', R, env)                      # [B,M,N]
    c = smooth_clamp(theta_stat[:, :, None] + mod)               # [B,M,N]
    out = {}
    for m, (name, kind, lo, hi) in enumerate(dest_specs(sr)):
        cm = c[:, m, :]
        if kind == "lin":
            out[name] = lo + cm * (hi - lo)
        else:  # log
            out[name] = tf.exp(np.log(lo) + cm * (np.log(hi) - np.log(lo)))
    return out


def oscillator(p, theta_osc, f0, N, sr, n_harm_saw, n_harm_sqr):
    """§2.3 morph オシレータ。p: modulate() の dict、theta_osc:[B,2]、f0:[B]"""
    out = 0.0
    levels = (p["level1"], p["level2"])
    fins = (p["fin1"], p["fin2"])
    for i in range(2):
        f_inst = f0[:, None] * tf.pow(2.0, fins[i] / 1200.0)     # [B,N]
        phi = tf.cumsum(f_inst, axis=1) / sr                     # [B,N]
        # saw: Σ_h (−1)^{h+1}/h sin(2π h φ)
        hs = tf.range(1, n_harm_saw + 1, dtype=tf.float32)       # [H]
        signs = tf.pow(-1.0, hs + 1.0)
        saw = tf.reduce_sum(
            (signs / hs)[None, None, :] *
            tf.sin(2.0 * np.pi * phi[:, :, None] * hs[None, None, :]), axis=-1)
        # square: Σ_j 1/(2j+1) sin(2π(2j+1)φ)
        js = tf.range(0, n_harm_sqr, dtype=tf.float32)
        odd = 2.0 * js + 1.0
        sqr = tf.reduce_sum(
            (1.0 / odd)[None, None, :] *
            tf.sin(2.0 * np.pi * phi[:, :, None] * odd[None, None, :]), axis=-1)
        a_i = theta_osc[:, i:i + 1]                              # [B,1]
        out = out + levels[i] * (a_i * saw + (1.0 - a_i) * sqr)
    return out


def svf_filter(x, cutoff, drive_f, q, sr):
    """§2.4(a) tanh drive → SVF(TPT)。x,cutoff,drive_f,q: [B,N]"""
    u = tf.tanh(drive_f * x)
    g = tf.tan(np.pi * cutoff / sr)
    k = 1.0 / q
    elems = (tf.transpose(u), tf.transpose(g), tf.transpose(k))  # [N,B]
    B = tf.shape(x)[0]
    init = (tf.zeros([B]), tf.zeros([B]), tf.zeros([B]))         # ic1, ic2, y

    def step(state, inp):
        ic1, ic2, _ = state
        un, gn, kn = inp
        a1 = 1.0 / (1.0 + gn * (gn + kn))
        a2 = gn * a1
        a3 = gn * a2
        v3 = un - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        return (2.0 * v1 - ic1, 2.0 * v2 - ic2, v2)

    scanned = tf.scan(step, elems, initializer=init)
    return tf.transpose(scanned[2])                              # [B,N]


def chorus(x, rate, depth, feedback, sr):
    """§2.4(b) 変調ディレイ+FB。depth はサンプル単位 [B,N]"""
    psi = tf.cumsum(rate, axis=1) / sr
    lfo = 0.5 * (1.0 + tf.sin(2.0 * np.pi * psi))
    D0 = D0_MS * 1e-3 * sr
    D = D0 + depth * lfo                                         # [B,N] samples
    L = int(2 ** np.ceil(np.log2(D0 + 0.010 * sr + 4)))          # ring buffer len
    B = tf.shape(x)[0]
    elems = (tf.transpose(x), tf.transpose(D), tf.transpose(feedback))
    init = (tf.zeros([B, L]), tf.zeros([B]))                     # buf, y

    def step(state, inp):
        buf, _ = state
        xn, Dn, fbn = inp
        Dn = tf.clip_by_value(Dn, 1.0, float(L - 2))
        i = tf.cast(tf.floor(Dn), tf.int32)
        f = Dn - tf.floor(Dn)
        idx1 = L - i                                             # b(n-i)   at buf[:,L-i]
        idx2 = L - i - 1
        w = (1.0 - f) * tf.gather(buf, tf.clip_by_value(idx1, 0, L - 1), batch_dims=1) \
            + f * tf.gather(buf, tf.clip_by_value(idx2, 0, L - 1), batch_dims=1)
        b_new = xn + fbn * w
        y = (1.0 - BETA_C) * xn + BETA_C * w
        buf = tf.concat([buf[:, 1:], b_new[:, None]], axis=1)
        return (buf, y)

    scanned = tf.scan(step, elems, initializer=init)
    return tf.transpose(scanned[1])


def compressor(x, thresh_db, ratio, g_mk_db, sr):
    """§2.4(c) 追従→dB→softニー→ゲイン+makeup。g_mk_db: [B]"""
    alpha = float(np.exp(-1.0 / (TAU_COMP * sr)))
    B = tf.shape(x)[0]
    elems = (tf.transpose(x), tf.transpose(thresh_db), tf.transpose(ratio))
    init = (tf.zeros([B]), tf.zeros([B]))                        # ell, y

    def step(state, inp):
        ell, _ = state
        xn, Tn, rho = inp
        ell = alpha * ell + (1.0 - alpha) * tf.abs(xn)
        L_db = 20.0 * tf.math.log(ell + EPS_LOG) / np.log(10.0)
        over = tf.math.softplus(GAMMA_KNEE * (L_db - Tn)) / GAMMA_KNEE
        G_db = -(1.0 - 1.0 / rho) * over
        y = tf.pow(10.0, (G_db + 0.0) / 20.0) * xn               # makeup は外で一括
        return (ell, y)

    scanned = tf.scan(step, elems, initializer=init)
    y = tf.transpose(scanned[1])
    return y * tf.pow(10.0, g_mk_db[:, None] / 20.0)


def _rbj_coeffs(kind, gain_db, f_c, sr, q=Q_MID, S=1.0):
    """RBJ Cookbook 係数（時変ゲイン対応）。gain_db: [B,N] → 各係数 [B,N]（a0 正規化済み）"""
    A = tf.pow(10.0, gain_db / 40.0)
    w0 = 2.0 * np.pi * f_c / sr
    cosw, sinw = float(np.cos(w0)), float(np.sin(w0))
    if kind == "peak":
        alpha = sinw / (2.0 * q)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cosw * tf.ones_like(A)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cosw * tf.ones_like(A)
        a2 = 1.0 - alpha / A
    else:
        sqA = tf.sqrt(A)
        alpha = sinw / 2.0 * float(np.sqrt((1.0 + 1.0) * (1.0 / S - 1.0) + 2.0)) \
            if False else sinw / 2.0 * tf.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)
        if kind == "lowshelf":
            b0 = A * ((A + 1) - (A - 1) * cosw + 2 * sqA * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
            b2 = A * ((A + 1) - (A - 1) * cosw - 2 * sqA * alpha)
            a0 = (A + 1) + (A - 1) * cosw + 2 * sqA * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cosw)
            a2 = (A + 1) + (A - 1) * cosw - 2 * sqA * alpha
        else:  # highshelf
            b0 = A * ((A + 1) + (A - 1) * cosw + 2 * sqA * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
            b2 = A * ((A + 1) + (A - 1) * cosw - 2 * sqA * alpha)
            a0 = (A + 1) - (A - 1) * cosw + 2 * sqA * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cosw)
            a2 = (A + 1) - (A - 1) * cosw - 2 * sqA * alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def eq3(x, low_db, mid_db, high_db, sr):
    """§2.4(d) 3バンド直列 biquad（TDF-II、時変係数を scan）"""
    bands = [_rbj_coeffs("lowshelf", low_db, F_LOW, sr),
             _rbj_coeffs("peak",     mid_db, F_MID, sr),
             _rbj_coeffs("highshelf", high_db, F_HIGH, sr)]
    # elems: [N,B,15]
    coef = tf.stack([tf.transpose(c) for band in bands for c in band], axis=-1)
    B = tf.shape(x)[0]
    init = (tf.zeros([B, 6]), tf.zeros([B]))                     # z(3band×2), y

    def step(state, inp):
        z, _ = state
        xn, cf = inp
        y = xn
        z_new = []
        for bi in range(3):
            b0, b1, b2, a1, a2 = [cf[:, 5 * bi + j] for j in range(5)]
            z1, z2 = z[:, 2 * bi], z[:, 2 * bi + 1]
            out = b0 * y + z1
            z1n = b1 * y - a1 * out + z2
            z2n = b2 * y - a2 * out
            z_new += [z1n, z2n]
            y = out
        return (tf.stack(z_new, axis=1), y)

    scanned = tf.scan(step, (tf.transpose(x), coef), initializer=init)
    return tf.transpose(scanned[1])


def distortion(x, drive_d, mix_d):
    """§2.4(e) tanh 波形整形（要素ごと）"""
    wet = tf.tanh(drive_d * x) / tf.tanh(drive_d)
    return (1.0 - mix_d) * x + mix_d * wet


def synth(theta_stat, theta_osc, g_mk_norm, env_params, R, f0, t_off,
          N, sr, n_harm_saw, n_harm_sqr, src_type=None):
    """順方向 g（§2.3 チェーン: osc→filter→chorus→comp→EQ→dist）
    theta_stat:[B,M](σ前空間) theta_osc:[B,2] g_mk_norm:[B]∈[0,1]
    env_params:[B,K,5] R:[B,K,M] f0:[B] → x:[B,N]"""
    env = sources(env_params, src_type, t_off, N, sr)            # [B,K,N]
    p = modulate(theta_stat, R, env, sr)
    x = oscillator(p, theta_osc, f0, N, sr, n_harm_saw, n_harm_sqr)
    x = svf_filter(x, p["cutoff"], p["drive_f"], p["res_q"], sr)
    x = chorus(x, p["rate"], p["depth"], p["feedback"], sr)
    g_mk_db = GMK_LO + g_mk_norm * (GMK_HI - GMK_LO)
    x = compressor(x, p["thresh"], p["ratio"], g_mk_db, sr)
    x = eq3(x, p["eq_low"], p["eq_mid"], p["eq_high"], sr)
    x = distortion(x, p["drive_d"], p["mix_d"])
    return x


def mrstft_loss(x, y, fft_sizes=(256, 512, 1024)):
    """§5.1 multi-res STFT（線形+対数）。x,y: [B,N] → [B]"""
    total = tf.zeros([tf.shape(x)[0]])
    for fl in fft_sizes:
        hop = fl // 4
        Sx = tf.abs(tf.signal.stft(x, fl, hop, fft_length=fl))
        Sy = tf.abs(tf.signal.stft(y, fl, hop, fft_length=fl))
        total += tf.reduce_mean(tf.abs(Sx - Sy), axis=[1, 2])
        total += tf.reduce_mean(tf.abs(tf.math.log(Sx + 1e-5) - tf.math.log(Sy + 1e-5)),
                                axis=[1, 2])
    return total
