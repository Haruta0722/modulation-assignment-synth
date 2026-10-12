r"""perceptual_dist.py — 局所的な JTFS による知覚距離（聞き分けの限界の較正に使う物差し）

JTFS（PNP の設定 J=8, Q=6, J_fr=2, Q_fr=2、折り返しを防ぐ）の 1 次（スカログラム）または 2 次を、時間方向に
全体平均せずに使う。
  1. 係数のパワー（振幅の 2 乗）を時間方向にガウス窓（標準偏差 sigma_ms）で平滑化し、振幅に戻す
     （干渉による速い揺れを打ち消し、耳の時間的な積分の代わりにする）
  2. その音の最大値から floor_db 下を下限にして対数をとる（聞こえないほど小さい値を無視する）
  3. 周波数（2 次ではチャネルも）で平均した差をフレームごとに出し、時間方向に agg（max / p90）でまとめる
     （時間全体で平均すると、音の頭だけの変化が薄まるため）
時間の形（音量の attack / release ±25%）と音量（0.5〜1 dB）の 2 つの弁別閾が同じ距離にそろうように選んだ（実験 34・35）。
"""
import numpy as np
import tensorflow as tf

from jtfs_tf import JTFS

HOP_MS = 8.0                                         # JTFS のスカログラムのフレーム間隔（128 / 16000 秒）


def gauss_smooth(a, sigma):
    """最後の軸（時間）をガウス窓（標準偏差 sigma フレーム）で平滑化"""
    if sigma <= 0:
        return a
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * (a.ndim - 1) + [(r, r)]
    ap = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += w * ap[..., i:i + a.shape[-1]]
    return out


class LocalJTFS:
    def __init__(self, sr=16000, order=2, sigma_ms=32, floor_db=30, agg="max"):
        self.order, self.sigma, self.floor_db, self.agg = order, sigma_ms / HOP_MS, floor_db, agg
        self.phi = JTFS(sr=sr, J=8, Q=6, J_fr=2, Q_fr=2, antialias=True)
        self._f = tf.function(self._raw, reduce_retracing=True)

    @classmethod
    def from_name(cls, name, sr=16000):
        """'u2/P32ms/30dB/max' のような名前から作る"""
        rep, sg, fl, agg = name.split("/")
        return cls(sr, order=int(rep[1]), sigma_ms=int(sg[1:-2]), floor_db=int(fl[:-2]), agg=agg)

    def _raw(self, x):
        u1 = self.phi.scalogram(x)
        if self.order == 1:
            return u1
        u2 = [u for psi_t, _ in self.phi.f2_t for psi_f, _ in self.phi.f2_f
              for u in self.phi._conv2d_c(u1, psi_t, psi_f)]
        return tf.stack(u2, axis=1)

    def features(self, x):
        """x: [B,N] → 平滑化した振幅 [B,(C,)F,T]（量子化していない音を渡すこと）"""
        a = self._f(tf.convert_to_tensor(x, tf.float32)).numpy()
        return np.sqrt(gauss_smooth(a ** 2, self.sigma))

    def distance(self, F0, F1):
        """F0 を基準に下限を決め、フレームごとの対数の差を時間方向にまとめる → [B]"""
        fax = tuple(range(1, F0.ndim - 1))
        peak = F0.max(axis=tuple(range(1, F0.ndim)), keepdims=True)
        e = peak * 10 ** (-self.floor_db / 20)
        dt = np.abs(np.log(F1 + e) - np.log(F0 + e)).mean(fax)                   # [B,T]
        return dt.max(1) if self.agg == "max" else np.percentile(dt, 90, axis=1)
