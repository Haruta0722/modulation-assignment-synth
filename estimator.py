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


def gumbel_st(logits, temp=1.0, hard=True, noise=True):
    """straight-through Gumbel-Softmax（forward one-hot / backward soft）

    noise=False で Gumbel ノイズを加えない = 決定的な最尤選択。
    **推論・評価時は必ず noise=False を用いる**。ノイズを残すと
    「分布からのサンプリング」になり、最尤の予測にならない
    （ノイズのスケールは温度に依存しないため temp を下げても消えない）。"""
    if noise:
        g = -tf.math.log(-tf.math.log(
            tf.random.uniform(tf.shape(logits), 1e-9, 1.0)))
    else:
        g = tf.zeros_like(logits)
    y = tf.nn.softmax((logits + g) / temp, axis=-1)
    if hard:
        y_h = tf.one_hot(tf.argmax(y, axis=-1), tf.shape(logits)[-1], dtype=y.dtype)
        y = tf.stop_gradient(y_h - y) + y
    return y


MOD_NFFT, MOD_BINS, MOD_GROUP, MOD_CH = 512, 82, 4, 32
RATE_NB, RATE_SIG, RATE_WIN = 128, 0.75, 3       # レートの分類（実験 44）: 区間の数、正解のぼかし幅（区間）、読み出しの窓（±区間）


def rate_from_logits(logits):
    """HL-Gauss の分類（実験 44）→ レート [Hz]。対数レート 0.1〜10 Hz を 128 区間に分けた確率のうち、
    確率が最大の区間の ±3 区間で重み付き平均を取る（区間の幅 約 3.7% より細かく読む）"""
    p = tf.nn.softmax(logits, axis=-1)
    j = tf.argmax(p, axis=-1, output_type=tf.int32)
    ar = tf.range(RATE_NB, dtype=tf.int32)
    w = p * tf.cast(tf.abs(ar - j[..., None]) <= RATE_WIN, tf.float32)
    c = (tf.cast(ar, tf.float32) + 0.5) / RATE_NB
    y = tf.reduce_sum(w * c, -1) / tf.maximum(tf.reduce_sum(w, -1), 1e-12)
    return 0.1 * tf.pow(100.0, y)


def hl_gauss_target(rate):
    """正解のレート [Hz] → 128 区間の確率。正解の位置を中心に、幅 0.75 区間のガウスでぼかす（HL-Gauss）"""
    y = tf.clip_by_value(tf.math.log(tf.maximum(rate, 1e-3) / 0.1) / np.log(100.0), 0.0, 1.0)
    edges = tf.linspace(0.0, 1.0, RATE_NB + 1)
    cdf = 0.5 * (1.0 + tf.math.erf((edges - y[..., None]) / (RATE_SIG / RATE_NB * np.sqrt(2.0))))
    p = cdf[..., 1:] - cdf[..., :-1]
    return p / tf.maximum(tf.reduce_sum(p, -1, keepdims=True), 1e-12)


def mod_spectrum(mel, phase=False):
    """変調スペクトル（実験 42）: 対数メル [B,T,64] の各帯域の時間変化を周波数分析 → [B,82,16]。
    帯域ごとに時間平均を引いてハン窓をかけ、512 点に 0 を詰めてフーリエ変換した振幅。
    62.5 コマ/秒なので 0.122 Hz 刻み。1〜82 段 = 0.12〜10.0 Hz（LFO のレートの範囲）を取り、
    隣り合う 4 帯域ずつ平均して 16 組にし、log(1 + 振幅) にする。
    phase=True（実験 46）: 位相も渡す。各帯域の複素係数を単位ベクトル（cos, sin）にして log(1 + 振幅) で重みを付け、
    4 帯域ずつ平均した実部・虚部を加える → [B,82,48]（振幅 16 組 ＋ 実部 16 組 ＋ 虚部 16 組）"""
    T = tf.shape(mel)[1]
    x = mel - tf.reduce_mean(mel, axis=1, keepdims=True)
    x = x * tf.signal.hann_window(T, periodic=False)[None, :, None]
    X = tf.signal.rfft(tf.transpose(x, [0, 2, 1]), fft_length=[MOD_NFFT])[:, :, 1:MOD_BINS + 1]   # [B,64,82] 複素
    amp = tf.abs(X)
    nb = mel.shape[2] // MOD_GROUP
    grp = lambda a: tf.reduce_mean(tf.reshape(a, [-1, nb, MOD_GROUP, MOD_BINS]), axis=2)     # [B,16,82]
    feats = [tf.math.log(1.0 + grp(amp))]
    if phase:
        w = tf.math.log(1.0 + amp) / tf.maximum(amp, 1e-6)
        feats += [grp(tf.math.real(X) * w), grp(tf.math.imag(X) * w)]
    return tf.transpose(tf.concat(feats, axis=1), [0, 2, 1])                          # [B,82,16 or 48]


def scale_grad(x, a):
    """値はそのまま通し、逆向きの勾配だけ a 倍にする（実験 45）。a = 0 で勾配を止める"""
    if a == 1.0:
        return x
    g = tf.stop_gradient(x)
    return g if a == 0.0 else g + a * (x - g)


def fixed_type(B):
    """typed の枠並び（実験 41）: 枠ごとに決まった型の one-hot [B,K,2]（class0=ENV）と、それに合う大きなロジット"""
    st = tf.constant(ds.SLOT_TYPE)
    oh = tf.tile(tf.stack([st, 1.0 - st], -1)[None], [B, 1, 1])
    return oh, 20.0 * (2.0 * oh - 1.0)


def leaky_relu(x):
    """負の入力にも傾き 0.01 を残す ReLU（働かなくなったユニットにも勾配が届く）"""
    return tf.nn.leaky_relu(x, alpha=0.01)


ACTIVATIONS = {"relu": "relu", "leaky": leaky_relu}


class Estimator(tf.keras.Model):
    def __init__(self, sr=16000, t_off=1.0, arch="v2", act="relu", lfo_read="trunk", rate_head="reg",
                 lfo_grad=1.0, mod_phase=False, rate_ce_stop=False, rate_aux=False):
        super().__init__()
        self.sr, self.t_off, self.arch, self.act = sr, t_off, arch, act
        self.rate_head = rate_head                # "reg": 対数レートの回帰（2 乗誤差） / "hlgauss": 区間の分類（実験 44）
        self.lfo_grad = lfo_grad                  # LFO のヘッドの損失が共有の層（trunk 以前）へ流す勾配の倍率（実験 45）
        self.mod_phase = mod_phase                # 変調スペクトルに位相も加える（実験 46）
        self.rate_ce_stop = rate_ce_stop          # レートの分類の勾配だけ共有の層に流さない（実験 46）
        self.rate_aux = rate_aux                  # レートの回帰（2 乗誤差）を補助として加え、共有の層まで流す（実験 46）
        if (rate_ce_stop or rate_aux) and rate_head != "hlgauss":
            raise ValueError("rate_ce_stop・rate_aux は rate_head=hlgauss のときのみ")
        if (rate_ce_stop or mod_phase) and lfo_read != "mod":
            raise ValueError("rate_ce_stop・mod_phase は lfo_read=mod のときのみ")
        if lfo_grad != 1.0 and lfo_read == "ends":
            raise ValueError("lfo_grad は lfo_read が trunk か mod のときのみ")
        self.lfo_read = lfo_read                  # "ends": LFO のヘッドだけ BiGRU の両端の状態から読む（v2〜v4）
                                                  # "mod" : LFO のヘッドに変調スペクトルの枝を加える（実験 42）
        a = ACTIVATIONS[act]                      # 畳み込み・時間方向の畳み込み・trunk の活性化関数
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(64, 513, sr, 30.0, 7800.0)
        if arch in ("v2", "v3", "v4"):
            # v8.2: 時間軸保存エンコーダ。周波数側を強くプーリング → BiGRU → まとめ
            #   v2: attention pooling（時間方向の重み付き平均）
            #   v3: 時間方向の畳み込みで 23 → 6 コマに縮めて並べる（重み付き平均では ENV を学べない、実験 15・16）
            #   v4: v2 と v3 を並べる（v3 では音全体の平均から分かる θ_stat・検出が下がった、実験 19）
            self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation=a)
                          for c, st in ((32, (1, 2)), (64, (2, 2)),
                                        (96, (1, 2)), (128, (2, 2)))]
            self.bigru = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(128, return_sequences=True))
            if arch in ("v2", "v4"):
                self.att = tf.keras.layers.Dense(1)
            if arch in ("v3", "v4"):
                self.tconv = [tf.keras.layers.Conv1D(64, 3, 2, "same", activation=a)
                              for _ in range(2)]
                self.drop = tf.keras.layers.Dropout(0.3)
        else:                                     # v1（最小骨格、比較用）
            self.convs = [tf.keras.layers.Conv2D(c, 3, 2, "same", activation=a)
                          for c in (32, 64, 128)]
            self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.trunk = tf.keras.layers.Dense(256, activation=a)
        if lfo_read == "ends":
            # 実験 37: 時間方向に平均してから位相を出すと、崩れた入力からは位相を学べない。
            # 順方向の最後の状態と逆方向の最後の状態（= 時刻 0 の位置）を、LFO 専用の層に通す
            self.lfo_trunk = tf.keras.layers.Dense(256, activation=a)
        if lfo_read == "mod":
            # 実験 42: LFO のレートを、変調スペクトル（各帯域の音量の揺れを時間方向に周波数分析したもの）の
            # 山の位置から読む。揺れの速さの方向には縮めずに畳み込み、trunk の出力と合わせて LFO 専用の層に通す
            self.mod_convs = [tf.keras.layers.Conv1D(MOD_CH, 5, padding="same", activation=a)
                              for _ in range(2)]
            self.mod_dense = tf.keras.layers.Dense(256, activation=a)
            self.lfo_trunk = tf.keras.layers.Dense(256, activation=a)
        d = tf.keras.layers.Dense
        self.h_env = d(K * 5)
        self.h_lfo = d(K * 4)
        if rate_head == "hlgauss":
            self.h_rate = d(K * RATE_NB)
        self.h_type = d(K * 2)
        self.h_gate = d(M * (K + 1))
        self.h_depth = d(M)
        self.h_stat = d(M)
        self.h_static = d(3)

    def encode(self, x, f0, training=False):
        return self._encode(x, f0, training)[0]

    def _encode(self, x, f0, training=False):
        """→ (trunk の出力, LFO 専用の特徴 or None)"""
        ends = rate_in = None
        S = tf.abs(tf.signal.stft(x, 1024, 256, fft_length=1024))     # [B,T,513]
        mel = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)          # [B,T,64]
        if self.lfo_read == "mod":
            m = mod_spectrum(mel, self.mod_phase)                      # [B,82,16 or 48]
            for c in self.mod_convs:
                m = c(m)                                               # [B,82,32]
            mod_feat = self.mod_dense(tf.reshape(m, [tf.shape(m)[0], MOD_BINS * MOD_CH]))
        h = mel[..., None]
        for c in self.convs:
            h = c(h)
        if self.arch in ("v2", "v3", "v4"):
            B = tf.shape(h)[0]
            h = tf.reshape(h, [B, -1, h.shape[2] * h.shape[3]])       # [B,T',F'·C]
            g = self.bigru(h)                                          # [B,T',256]
            if self.lfo_read == "ends":
                ends = self.lfo_trunk(tf.concat([g[:, -1, :128], g[:, 0, 128:]], -1))
            parts = []
            if self.arch in ("v2", "v4"):                              # 音全体の平均
                w = tf.nn.softmax(self.att(g), axis=1)
                parts += [tf.reduce_sum(w * g, axis=1), tf.reduce_mean(g, axis=1)]   # [B,512]
            if self.arch in ("v3", "v4"):                              # 時間の形
                t = g
                for c in self.tconv:
                    t = c(t)                                           # [B,6,64]
                parts.append(self.drop(tf.reshape(t, [B, t.shape[1] * t.shape[2]]),
                                       training=training))             # [B,384]
            h = tf.concat(parts, axis=-1)
        else:
            h = self.gap(h)
        f0n = tf.math.log(f0 / F0_LO)[:, None] / np.log(16.0)         # MIDI36-84 → [0,1]
        out = self.trunk(tf.concat([h, f0n], axis=-1))
        if self.lfo_read == "mod":
            ends = self.lfo_trunk(tf.concat([scale_grad(out, self.lfo_grad), mod_feat], axis=-1))
            if self.rate_ce_stop:
                # 実験 46: 同じ LFO 専用の層に、勾配を止めた trunk の出力をもう一度通してレートの分類だけに使う。
                # 分類の勾配は LFO 専用の層・変調スペクトルの枝には流れ、共有の層には流れない
                rate_in = self.lfo_trunk(tf.concat([tf.stop_gradient(out), mod_feat], axis=-1))
        return out, ends, rate_in

    def call(self, x, f0, temp=1.0, hard=True, noise=True, training=False):
        B = tf.shape(x)[0]
        h, h_l, h_r = self._encode(x, f0, training=training)
        if h_l is None:
            h_l = scale_grad(h, self.lfo_grad)
        if h_r is None:
            h_r = h_l
        # --- ENV（割合パラメータ化） ---
        ue = tf.sigmoid(tf.reshape(self.h_env(h), [B, K, 5]))
        a = tf.maximum(ue[..., 0] * self.t_off, 1e-3)
        hh = ue[..., 1] * (self.t_off - a)
        dd = tf.maximum(ue[..., 2] * (self.t_off - a - hh), 1e-3)
        s = ue[..., 3]
        r = ue[..., 4] * R_MAX + 1e-3
        p_env = tf.stack([a, hh, dd, s, r], axis=-1)                  # [B,K,5]
        # --- LFO（rate 対数 / 位相は (cos,sin) → 円環復元） ---
        ul = tf.reshape(self.h_lfo(h_l), [B, K, 4])
        if self.rate_head == "hlgauss":                               # 実験 44: レートは区間の分類で出す
            rate_logits = tf.reshape(self.h_rate(h_r), [B, K, RATE_NB])
            rate = rate_from_logits(rate_logits)
        else:
            rate = 0.1 * tf.pow(100.0, tf.sigmoid(ul[..., 0]))
        phase = tf.math.floormod(tf.atan2(ul[..., 2], ul[..., 1]) / (2 * np.pi), 1.0)
        morph = tf.sigmoid(ul[..., 3])
        zeros = tf.zeros_like(rate)
        p_lfo = tf.stack([rate, phase, morph, zeros, zeros], axis=-1)
        # --- タイプ（ST-Gumbel: forward one-hot → パラメータ選択が厳密） ---
        if ds.SLOT_TYPE is not None:                                  # typed: 型は枠で決まる（実験 41）
            type_oh, type_logits = fixed_type(B)
        else:
            type_logits = tf.reshape(self.h_type(h), [B, K, 2])       # class0=ENV
            type_oh = gumbel_st(type_logits, temp, hard, noise)
        w_env = type_oh[..., 0]                                       # [B,K]
        src_params = w_env[..., None] * p_env + (1 - w_env[..., None]) * p_lfo
        # --- ゲート・深さ ---
        gate_logits = tf.reshape(self.h_gate(h), [B, M, K + 1])
        gate_oh = gumbel_st(gate_logits, temp, hard, noise)           # [B,M,K+1]
        depth = tf.tanh(self.h_depth(h))                              # [B,M]
        R_hat = tf.transpose(gate_oh[..., :K], [0, 2, 1]) * depth[:, None, :]  # [B,K,M]
        # --- θ_stat（fin 正準順序を構造保証） ---
        us = tf.sigmoid(self.h_stat(h))                               # [B,M]
        f1 = us[:, 2] * 0.95                                          # 上限余白
        f2 = f1 + 0.05 + us[:, 3] * (1.0 - f1 - 0.05)
        theta = tf.concat([us[:, :2], f1[:, None], f2[:, None], us[:, 4:]], axis=1)
        # --- 静的のみ ---
        st = tf.sigmoid(self.h_static(h))
        out = dict(src_params=src_params, type_logits=type_logits, type_oh=type_oh,
                   gate_logits=gate_logits, gate_oh=gate_oh, depth=depth,
                   R=R_hat, theta_stat=theta, theta_osc=st[:, :2], gmk=st[:, 2])
        if self.rate_head == "hlgauss":
            out["rate_logits"] = rate_logits
        if self.rate_aux:                                             # 補助のレートの回帰（予測には使わない）
            out["rate_reg"] = 0.1 * tf.pow(100.0, tf.sigmoid(ul[..., 0]))
        return out


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


def env_frac(p, t_off):
    """ENV の (a,h,d,s,r)[秒] → 生成・出力と同じ割合空間 (a/T, h/残り, d/残り, s, r/R_MAX)。
    秒で比べると hold・decay は前段の誤差を引き継ぎ、尺度も小さく偏るため、割合で比べる"""
    a, h, d, s, r = tf.unstack(p, axis=-1)
    u0 = a / t_off
    u1 = h / tf.maximum(t_off - a, 1e-3)
    u2 = d / tf.maximum(t_off - a - h, 1e-3)
    u = tf.stack([u0, u1, u2, s, r / R_MAX], axis=-1)
    return tf.clip_by_value(u, 0.0, 2.0)                              # LFO の枠の値で発散させない


def teacher_R(depth, z):
    """深さを真の割り当て z（-1=∅）の位置に置いた R。深さの教師をゲートの当たり外れから切り離す。
    予測 R（ゲート×深さ）で教えると、ゲートが外れた学習例で深さが 0 に引かれ、p(正解)·d に縮む"""
    oh = tf.one_hot(tf.cast(z, tf.int32), K)                          # [B,M,K]、-1 は全 0
    return tf.transpose(oh, [0, 2, 1]) * depth[:, None, :]            # [B,K,M]


def none_depth_loss(depth, z):
    """真が ∅ の列で深さを 0 に教える項（teacher_R ではその列に勾配が流れないため）"""
    off = tf.cast(tf.less(tf.cast(z, tf.int32), 0), tf.float32)       # [B,M]
    return tf.reduce_mean(tf.reduce_sum(off * depth ** 2, -1))


def losses(pred, batch, x_true, sr, t_off, n_harm, lam, matching=True):
    # v8.3: used[B,K]=そのソースがどこかに配線されているか（聞こえるか）
    f0, true_R = batch["f0"], batch["R"]
    true_type = batch["src_type"]                                     # 1=ENV
    if matching:
        mapping = match_perm(pred, true_R, true_type)
    else:                                             # v8.2: 正準順序データでは恒等
        B = tf.shape(true_R)[0]
        mapping = tf.tile(tf.range(K, dtype=tf.int32)[None, :], [B, 1])

    src_p = gather_rows(pred["src_params"], mapping)                  # [B,K,5]
    type_lg = gather_rows(pred["type_logits"], mapping)
    R_hat = (teacher_R(pred["depth"], batch["z"]) if lam.get("depth_tf", 0)
             else gather_rows(pred["R"], mapping))

    # --- param 損失: λI 版（等方 L2）／対角知覚計量版（v8.8）を lam["mdiag"] で切替 ---
    # mdiag: [P] の知覚感度ベクトル（precompute_Mdiag.py の d_mean）。None なら等方。
    md = lam.get("mdiag", None)
    def _w(sl):                      # θ_all の該当スライスの重み（[..]）
        if md is None:
            return 1.0
        return tf.constant(md[sl[0]:sl[1]], tf.float32)

    l_stat = tf.reduce_mean(tf.reduce_sum(
        _w((0, 17)) * (pred["theta_stat"] - batch["theta_stat"]) ** 2, -1))
    l_static = tf.reduce_mean(tf.reduce_sum(
        _w((17, 19)) * (pred["theta_osc"] - batch["theta_osc"]) ** 2, -1)
        + (1.0 if md is None else float(md[19])) * (pred["gmk"] - batch["gmk"]) ** 2)
    wR = 1.0 if md is None else tf.reshape(tf.constant(md[40:108], tf.float32), [K, M])
    l_R = tf.reduce_mean(tf.reduce_sum(wR * (R_hat - true_R) ** 2, [1, 2]))
    if lam.get("depth_none0", 0):
        l_R += none_depth_loss(pred["depth"], batch["z"])
    z_all = tf.cast(batch["z"], tf.int32)                             # [B,M]
    used = tf.reduce_any(tf.equal(z_all[:, :, None],
                                  tf.range(K, dtype=tf.int32)[None, None, :]),
                         axis=1)                                      # [B,K]
    uf = tf.cast(used, tf.float32)
    nu = tf.maximum(tf.reduce_sum(uf), 1.0)
    is_env = true_type                                                # [B,K]
    wS = (tf.ones([K, 5]) if md is None
          else tf.reshape(tf.constant(md[20:40], tf.float32), [K, 5]))
    if lam.get("env_frac", 0):
        ew = tf.constant(lam.get("env_w", [1.0] * 5), tf.float32)   # (a,h,d,s,r) ごとの重み
        d_env = tf.reduce_sum(
            ew * wS * (env_frac(src_p, t_off) - env_frac(batch["src_params"], t_off)) ** 2, -1)
    else:
        d_env = tf.reduce_sum(wS * ((src_p - batch["src_params"]) / SLOT_SCALE_ENV) ** 2, -1)
    dphi = tf.abs(src_p[..., 1] - batch["src_params"][..., 1])
    circ = tf.minimum(dphi, 1.0 - dphi)
    lr_n = (tf.math.log(tf.maximum(src_p[..., 0], 1e-3) / 0.1)
            - tf.math.log(tf.maximum(batch["src_params"][..., 0], 1e-3) / 0.1)) / np.log(100.0)
    if "rate_logits" in pred:                                         # 実験 44: レートは HL-Gauss の交差エントロピー
        r_term = tf.nn.softmax_cross_entropy_with_logits(
            labels=hl_gauss_target(batch["src_params"][..., 0]),
            logits=gather_rows(pred["rate_logits"], mapping))
        if "rate_reg" in pred:                                        # 実験 46: 補助のレートの回帰（共有の層まで流す）
            lr_a = (tf.math.log(tf.maximum(gather_rows(pred["rate_reg"], mapping), 1e-3) / 0.1)
                    - tf.math.log(tf.maximum(batch["src_params"][..., 0], 1e-3) / 0.1)) / np.log(100.0)
            r_term = r_term + lr_a ** 2
    else:
        r_term = lr_n ** 2
    d_lfo = (wS[:, 0] * r_term + wS[:, 1] * circ ** 2
             + wS[:, 2] * (src_p[..., 2] - batch["src_params"][..., 2]) ** 2)
    l_src = tf.reduce_sum(uf * (is_env * d_env + (1 - is_env) * d_lfo)) / nu
    l_param = l_stat + l_static + l_R + lam.get("src_w", 1.0) * l_src

    # --- ゲート/型 CE（π* でクラス添字を揃える） ---
    gate_lg = pred["gate_logits"]                                     # [B,M,K+1]
    inv = tf.argsort(mapping, axis=1)                                 # true k → pred 行
    z = tf.cast(batch["z"], tf.int32)                                 # [B,M] (-1=∅)
    z_re = tf.where(z >= 0, tf.gather(inv, tf.maximum(z, 0), axis=1, batch_dims=1), K)
    # v8.5: ∅ クラス（17列中 ~13.5列）の重みを下げ、CE 勾配の ∅ 支配を緩和する。
    #       none_w=1.0 で従来と同じ。lam["none_w"] で調整（既定 0.25）。
    ce_g = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=z_re, logits=gate_lg)
    w_none = lam.get("none_w", 1.0)
    wg = tf.where(tf.equal(z_re, K), w_none * tf.ones_like(ce_g), tf.ones_like(ce_g))
    l_gate = tf.reduce_sum(wg * ce_g) / tf.maximum(tf.reduce_sum(wg), 1e-6)
    t_lab = tf.cast(1.0 - true_type, tf.int32)                        # ENV→0, LFO→1
    ce_t = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=t_lab, logits=type_lg)                                 # [B,K]
    l_type = tf.reduce_sum(uf * ce_t) / nu                            # 未使用ソースは除外

    # --- 疎性（soft 確率で） ---
    p_assign = 1.0 - tf.nn.softmax(gate_lg, -1)[..., K]
    l_sparse = tf.reduce_mean(tf.reduce_sum(p_assign, -1))

    # --- スペクトル（再合成、ST one-hot で厳密分岐） ---
    if lam["spec"] <= 0.0:                       # 診断用: 再合成をスキップ
        l_spec = tf.constant(0.0)
        total = (lam["param"] * l_param + lam["assign"] * (l_gate + l_type)
                 + lam["sparse"] * l_sparse)
        return total, dict(spec=l_spec, param=l_param, gate=l_gate,
                           type=l_type, sparse=l_sparse), mapping
    x_hat = synth_fast(pred["theta_stat"], pred["theta_osc"], pred["gmk"],
                       pred["src_params"], pred["R"], f0, t_off,
                       int(x_true.shape[1]), sr, n_harm[0], n_harm[1],
                       src_type=pred["type_oh"][..., 0])
    l_spec = tf.reduce_mean(ds.mrstft_loss(x_hat, x_true))

    total = (lam["spec"] * l_spec + lam["param"] * l_param
             + lam["assign"] * (l_gate + l_type) + lam["sparse"] * l_sparse)
    return total, dict(spec=l_spec, param=l_param, gate=l_gate,
                       type=l_type, sparse=l_sparse), mapping
