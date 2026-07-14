"""
jtfs_tf.py — Joint Time-Frequency Scattering (JTFS) の TensorFlow 実装
================================================================================
仕様 v8 §5.2 の知覚表現 Φ。Kymatio 非依存（SciPy 互換問題の回避＋TF の
forward-mode JVP をそのまま使えるようにするため自前実装）。

構成:
  1) 1次: Morlet の定Qフィルタバンク（$Q_1$）で複素ウェーブレット変換 →
     絶対値 → スカログラム $U_1(t,\\log\\lambda)$（低域平均で時間ダウンサンプル）
  2) 2次: 時間-周波数の joint ウェーブレット
     $\\Psi_{\\alpha,\\beta,s}(t,\\log\\lambda)=\\psi_\\alpha(t)\\,\\psi_{\\beta,s}(\\log\\lambda)$
     （$s=\\pm1$ が up/down チャープの向き）を $U_1$ に 2D 畳み込み → 絶対値
  3) 時間平均（低域）→ 係数を連結して 1 本のベクトル $\\Phi(x)$

すべて実数演算（複素は実部/虚部の2チャネル）で書いてあり、TF の自動微分
（forward/reverse）が通る。
"""
import numpy as np
import tensorflow as tf


def morlet_1d(n, xi, sigma):
    """時間領域 Morlet（DC 補正あり）。n: 長さ, xi: 正規化中心周波数, sigma: 帯域"""
    t = np.arange(n) - n // 2
    gauss = np.exp(-t ** 2 / (2.0 * sigma ** 2))
    wave = np.exp(1j * 2 * np.pi * xi * t)
    kappa = gauss.sum() / max(np.abs(gauss * wave).sum(), 1e-12)
    psi = gauss * (wave - kappa)                     # DC を落とす
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) + 1e-12)
    return psi


def cqt_bank(J, Q, n_taps=None):
    """定Qフィルタバンク。J オクターブ × Q ボイス → [(psi, xi)]"""
    filters = []
    for j in range(J):
        for q in range(Q):
            xi = 0.4 * 2.0 ** (-(j + q / Q))         # 0.4 (Nyquist 手前) から下降
            sigma = 0.1 / xi                          # 定Q: sigma ∝ 1/xi
            n = n_taps or int(min(8 * sigma, 4096))
            n += n % 2                                # 偶数長
            filters.append((morlet_1d(n, xi, sigma).astype(np.complex64), xi))
    return filters


class JTFS(tf.keras.layers.Layer):
    """Φ(x): [B,N] → [B,D]（時間平均済みの JTFS 係数）"""

    def __init__(self, J=12, Q=8, J_fr=3, Q_fr=2, T_log2=11, sr=16000, **kw):
        super().__init__(**kw)
        self.sr = sr
        self.T = 2 ** T_log2                          # 時間平均窓（サンプル）
        # --- 1次: 時間 CQT ---
        self.f1 = cqt_bank(J, Q)
        self.n_lam = len(self.f1)                     # 周波数軸の長さ
        # --- 2次: joint (時間 × 周波数) ---
        self.f2_t = cqt_bank(J_fr, Q_fr, n_taps=256)          # 時間側（低レート）
        self.f2_f = cqt_bank(2, Q_fr, n_taps=min(2 * self.n_lam, 64))  # 周波数側
        self.ds = max(self.T // 16, 1)                # スカログラムの時間ダウンサンプル

    def _conv1d_c(self, x, psi):
        """x:[B,N] 実、psi: 複素 → |x*psi|: [B,N]"""
        kr = tf.constant(np.real(psi)[::-1, None, None].copy(), tf.float32)
        ki = tf.constant(np.imag(psi)[::-1, None, None].copy(), tf.float32)
        xe = x[:, :, None]
        yr = tf.nn.conv1d(xe, kr, 1, "SAME")
        yi = tf.nn.conv1d(xe, ki, 1, "SAME")
        return tf.sqrt(yr ** 2 + yi ** 2 + 1e-12)[:, :, 0]

    def scalogram(self, x):
        """U1: [B, n_lam, T'] （時間ダウンサンプル済み）"""
        rows = [self._conv1d_c(x, psi)[:, ::self.ds] for psi, _ in self.f1]
        return tf.stack(rows, axis=1)

    def _conv2d_c(self, u, psi_t, psi_f):
        """u:[B,F,T] に分離可能な複素 2D ウェーブレット（向き s=±1 の2種）"""
        outs = []
        for s in (1, -1):
            pf = psi_f if s > 0 else np.conj(psi_f)   # 向き（up/down チャープ）
            kt_r = tf.constant(np.real(psi_t)[::-1].copy(), tf.float32)
            kt_i = tf.constant(np.imag(psi_t)[::-1].copy(), tf.float32)
            kf_r = tf.constant(np.real(pf)[::-1].copy(), tf.float32)
            kf_i = tf.constant(np.imag(pf)[::-1].copy(), tf.float32)
            # 分離可能: (a+bi)⊗(c+di) の実部/虚部を 2 段の 1D conv で構成
            ue = u[:, :, :, None]                                    # [B,F,T,1]
            def cf(k):     # 周波数軸 conv
                kk = tf.reshape(k, [-1, 1, 1, 1])
                return tf.nn.conv2d(ue, kk, 1, "SAME")
            fr, fi = cf(kf_r), cf(kf_i)
            def ct(v, k):  # 時間軸 conv
                kk = tf.reshape(k, [1, -1, 1, 1])
                return tf.nn.conv2d(v, kk, 1, "SAME")
            rr = ct(fr, kt_r) - ct(fi, kt_i)
            ii = ct(fr, kt_i) + ct(fi, kt_r)
            outs.append(tf.sqrt(rr ** 2 + ii ** 2 + 1e-12)[..., 0])  # [B,F,T]
        return outs

    def call(self, x):
        u1 = self.scalogram(x)                                       # [B,F,T']
        # S1: 1次の時間平均
        s1 = tf.reduce_mean(u1, axis=2)                              # [B,F]
        coeffs = [s1]
        # S2: joint 2次
        for psi_t, xi_t in self.f2_t:
            for psi_f, xi_f in self.f2_f:
                for u2 in self._conv2d_c(u1, psi_t, psi_f):
                    coeffs.append(tf.reduce_mean(u2, axis=2))        # [B,F]
        return tf.concat(coeffs, axis=1)                             # [B,D]
