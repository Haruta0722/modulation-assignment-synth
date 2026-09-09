"""
two_stage.py — 二段構成: 軌跡生成 → パラメータ当てはめ
================================================================================
診断（diag_output.py）の結論:
  同一データ・同一エンコーダで出力表現だけを変えると
    free（フレーム列を直接出力）   shape_ok = 0.729
    param（ADHSR/LFO パラメータ経由） shape_ok = 0.333
  → **パラメータ化そのものがボトルネック**（型推定は type_acc=0.990 で成功している）

設計:
  Stage 1  音 → 軌跡 e(n) をフレーム列として自由に予測（制約なし）
           教師: 真の軌跡（信号空間の L1）
  Stage 2  軌跡 → ADHSR/LFO パラメータを当てはめる
           教師: (a) 真のパラメータ（あれば）, (b) 再構成軌跡と Stage1 出力の一致

  重要な設計判断:
    - Stage 2 の入力は**軌跡のみ**（音を見ない）。これにより
      「音→軌跡」と「軌跡→パラメータ」が分離され、誤差の波及が局所化する
    - Stage 2 は軽量（軌跡は 1 次元系列なので小さな GRU で足る）
    - 学習は同時（end-to-end）だが、Stage 1 の教師が直接あるため
      パラメータ化の誤差が Stage 1 に逆流しない（stop_gradient で遮断可）

比較対象:
  --mode free    : Stage1 のみ（上限の参照値）
  --mode param   : 単段パラメータ出力（diag_output の param 相当。下限の参照値）
  --mode two     : 二段構成（本手法）
  --detach       : Stage2 の勾配を Stage1 に流さない（誤差波及の遮断）

使い方:
  python two_stage.py --mode two --target 4 --steps 3000
  python two_stage.py --mode two --target 4 --steps 3000 --detach
  python two_stage.py --mode two --target 4 --n_mod 3 --steps 3000
"""
import argparse
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast
from diag_concurrent import make_conc
from diag_shape import F0_LO

K, M = ds.K_ENVS, ds.M_DEST
R_MAX = 0.6


class Stage1(tf.keras.Model):
    """音 → 軌跡（フレーム列）"""

    def __init__(self, sr, hop=256, mel=64):
        super().__init__()
        self.hop = hop
        self.mel_mat = tf.signal.linear_to_mel_weight_matrix(mel, 513, sr, 30., 7800.)
        self.convs = [tf.keras.layers.Conv2D(c, 3, st, "same", activation="relu")
                      for c, st in ((32, (1, 2)), (64, (1, 2)),
                                    (96, (1, 2)), (128, (1, 2)))]
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=True))
        self.head = tf.keras.layers.Dense(1)

    def call(self, x, f0):
        S = tf.abs(tf.signal.stft(x, 1024, self.hop, fft_length=1024))
        h = tf.math.log(tf.matmul(S, self.mel_mat) + 1e-5)[..., None]
        for c in self.convs:
            h = c(h)
        B, T = tf.shape(h)[0], tf.shape(h)[1]
        h = tf.reshape(h, [B, T, h.shape[2] * h.shape[3]])
        f0n = tf.math.log(f0 / F0_LO)[:, None] / np.log(16.0)
        h = tf.concat([h, f0n[:, None, :] * tf.ones([1, T, 1])], -1)
        return tf.sigmoid(self.head(self.bigru(h)))[..., 0]     # [B,T]


def traj_features(traj, t_off, sr, hop, n_frames):
    """軌跡から ADHSR の構造に対応する明示的特徴を抽出（第三の改良）"""
    T = tf.shape(traj)[1]
    tt = tf.cast(tf.range(T), tf.float32) * (hop / sr)          # フレーム時刻
    peak = tf.reduce_max(traj, 1, keepdims=True)
    amin = tf.reduce_min(traj, 1, keepdims=True)
    argmax_t = tf.gather(tt, tf.argmax(traj, 1))                # ピーク到達時刻
    # 立ち上がり: 最大値の 10%/90% を超える最初の時刻
    def first_cross(frac):
        thr = amin + frac * (peak - amin)
        above = tf.cast(traj >= thr, tf.float32)
        idx = tf.argmax(above, axis=1)
        return tf.gather(tt, idx)
    t10, t90 = first_cross(0.1), first_cross(0.9)
    # 定常値（ノートオフ直前の平均）と減衰
    n_off = tf.cast(tf.round(t_off * sr / hop), tf.int32)
    n_off = tf.minimum(n_off, T - 1)
    sus = tf.reduce_mean(traj[:, tf.maximum(n_off - 5, 0):n_off], 1)
    tail = tf.reduce_mean(traj[:, n_off:], 1)
    d1 = traj[:, 1:] - traj[:, :-1]
    return tf.stack([peak[:, 0], amin[:, 0], argmax_t, t10, t90, t90 - t10,
                     sus, tail, tf.reduce_mean(d1, 1), tf.math.reduce_std(d1, 1),
                     tf.reduce_mean(tf.abs(d1), 1),
                     tf.cast(tf.math.count_nonzero(d1 > 0, axis=1), tf.float32)
                     / tf.cast(T, tf.float32)], -1)              # [B,12]


class Stage2(tf.keras.Model):
    """軌跡 → ADHSR/LFO パラメータ（音は見ない）

    refine>0 で反復修正（第二の改良）: 初期推定 → 再構成 → 残差を見て修正、を繰り返す
    use_feat=True で明示的特徴を併用（第三の改良）
    """

    def __init__(self, t_off, refine=0, use_feat=False, sr=16000, hop=256):
        super().__init__()
        self.t_off, self.refine, self.use_feat = t_off, refine, use_feat
        self.sr, self.hop = sr, hop
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(64, return_sequences=True))
        self.pool = tf.keras.layers.Dense(128, activation="relu")
        self.h_env = tf.keras.layers.Dense(5)
        self.h_lfo = tf.keras.layers.Dense(4)
        self.h_type = tf.keras.layers.Dense(2)
        if refine > 0:      # 残差を見て修正量を出すヘッド（重み共有）
            self.r_gru = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(48, return_sequences=True))
            self.r_pool = tf.keras.layers.Dense(96, activation="relu")
            self.r_env = tf.keras.layers.Dense(5)
            self.r_lfo = tf.keras.layers.Dense(4)

    def _embed(self, traj):
        h = self.gru(traj[..., None])
        g = tf.concat([tf.reduce_mean(h, 1), tf.reduce_max(h, 1)], -1)
        if self.use_feat:
            n_frames = int(traj.shape[1]) if traj.shape[1] is not None else 0
            g = tf.concat([g, traj_features(traj, self.t_off, self.sr,
                                            self.hop, n_frames)], -1)
        return self.pool(g)

    def call(self, traj, type_oracle=None, temp=0.5):
        g = self._embed(traj)
        v_env, v_lfo = self.h_env(g), self.h_lfo(g)
        tl = self.h_type(g)
        p_env, p_lfo = self._to_params(v_env, v_lfo)
        if type_oracle is not None:
            w = type_oracle[:, None]
        else:
            gum = -tf.math.log(-tf.math.log(
                tf.random.uniform(tf.shape(tl), 1e-9, 1.0)))
            y = tf.nn.softmax((tl + gum) / temp, -1)
            yh = tf.one_hot(tf.argmax(y, -1), 2, dtype=y.dtype)
            w = (tf.stop_gradient(yh - y) + y)[:, 0:1]
        styp = tf.concat([w, tf.zeros_like(w)], -1)[:, :1]
        p = w[..., None] * p_env[:, None] + (1 - w[..., None]) * p_lfo[:, None]
        # --- 反復修正（第二）: 再構成と入力軌跡の残差を見て v を更新 ---
        for _ in range(self.refine):
            rec = self._render(p, styp, int(traj.shape[1]))
            resid = tf.stack([traj, rec, traj - rec], -1)        # [B,T,3]
            hr = self.r_gru(resid)
            gr = self.r_pool(tf.concat([tf.reduce_mean(hr, 1),
                                        tf.reduce_max(hr, 1)], -1))
            v_env = v_env + self.r_env(gr)
            v_lfo = v_lfo + self.r_lfo(gr)
            p_env, p_lfo = self._to_params(v_env, v_lfo)
            p = w[..., None] * p_env[:, None] + (1 - w[..., None]) * p_lfo[:, None]
        return p, styp, tl                                  # [B,1,5], [B,1], [B,2]

    def _to_params(self, v_env, v_lfo):
        ue = tf.sigmoid(v_env)
        a = tf.maximum(ue[:, 0] * self.t_off, 1e-3)
        hh = ue[:, 1] * (self.t_off - a)
        dd = tf.maximum(ue[:, 2] * (self.t_off - a - hh), 1e-3)
        p_env = tf.stack([a, hh, dd, ue[:, 3], ue[:, 4] * R_MAX + 1e-3], -1)
        rate = 0.5 * tf.pow(20.0, tf.sigmoid(v_lfo[:, 0]))
        phase = tf.math.floormod(
            tf.atan2(v_lfo[:, 2], v_lfo[:, 1]) / (2 * np.pi), 1.0)
        z = tf.zeros_like(rate)
        p_lfo = tf.stack([rate, phase, tf.sigmoid(v_lfo[:, 3]), z, z], -1)
        return p_env, p_lfo

    def _render(self, p, styp, n_frames):
        """パラメータ → 軌跡（フレーム）。反復修正の内部で使う"""
        N = int(round(self.t_off * 1.5 * self.sr))
        e = ds.sources(p, styp, self.t_off, N, self.sr)[:, 0]
        idx = tf.minimum(tf.range(n_frames) * self.hop + 512, N - 1)
        return tf.gather(e, idx, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["free", "param", "two"], default="two")
    ap.add_argument("--target", type=int, default=4)
    ap.add_argument("--n_mod", type=int, default=1)
    ap.add_argument("--param_sup", type=float, default=0.0,
                    help="第一: 真のパラメータを教師に加える重み（0=無効, 推奨 1.0）")
    ap.add_argument("--refine", type=int, default=0,
                    help="第二: Stage2 の反復修正回数（0=無効, 推奨 2〜3）")
    ap.add_argument("--use_feat", action="store_true",
                    help="第三: 軌跡から明示的特徴を抽出して併用")
    ap.add_argument("--tto", type=int, default=0,
                    help="推論時最適化（test-time optimization）の反復数。0=無効, 推奨 50")
    ap.add_argument("--tto_lr", type=float, default=0.05)
    ap.add_argument("--s2_input", choices=["stage1", "true", "noisy"],
                    default="stage1",
                    help="Stage2 の入力: stage1=通常 / true=真の軌跡（上限診断） / "
                         "noisy=真の軌跡+ノイズ（誤差耐性の測定）")
    ap.add_argument("--noise", type=float, default=0.1,
                    help="s2_input=noisy のときの付加ノイズ標準偏差")
    ap.add_argument("--detach", action="store_true",
                    help="Stage2 の勾配を Stage1 に流さない")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--mel", type=int, default=64)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--toff", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    SR, N, T_OFF = args.sr, int(args.sr * args.T), args.toff
    n_frames = 1 + (N - 1024) // args.hop
    fidx = tf.constant(np.minimum(np.arange(n_frames) * args.hop + 512, N - 1))
    fmax = F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)
    tname = ds.dest_specs(SR)[args.target][0]

    s1 = Stage1(SR, args.hop, args.mel)
    s2 = (Stage2(T_OFF, args.refine, args.use_feat, SR, args.hop)
          if args.mode != "free" else None)
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function(reduce_retracing=True)
    def synth(th, osc, gmk, src, R, f0, styp):
        return synth_fast(th, osc, gmk, src, R, f0, T_OFF, N, SR,
                          NH[0], NH[1], src_type=styp)

    def batch():
        th, osc, gmk, src, styp, R, f0, kt = make_conc(
            rng, args.batch, args.target, SR, N, T_OFF, args.n_mod)
        x = synth(*[tf.constant(v) for v in (th, osc, gmk, src, R, f0, styp)])
        e = ds.sources(tf.constant(src), tf.constant(styp), T_OFF, N, SR).numpy()
        ar = np.arange(len(kt))
        y = e[ar, kt][:, fidx.numpy()].astype(np.float32)
        return (x, tf.constant(f0), tf.constant(y),
                tf.constant(styp[ar, kt].astype(np.float32)),
                tf.constant(src[ar, kt].astype(np.float32)))

    def reconstruct(p, st):
        """パラメータ → 軌跡（フレーム）"""
        e = ds.sources(p, st, T_OFF, N, SR)[:, 0]
        return tf.gather(e, fidx, axis=1)

    def params_to_free(p, st):
        """パラメータ → 自由変数（sigmoid の逆写像）。TTO の初期値に使う"""
        eps = 1e-4
        def inv(v):
            v = tf.clip_by_value(v, eps, 1 - eps)
            return tf.math.log(v / (1 - v))
        a, h, d, sus, r = [p[:, 0, i] for i in range(5)]
        v_env = tf.stack([
            inv(a / T_OFF),
            inv(h / tf.maximum(T_OFF - a, eps)),
            inv(d / tf.maximum(T_OFF - a - h, eps)),
            inv(sus), inv(r / R_MAX)], -1)
        rate, ph, mo = p[:, 0, 0], p[:, 0, 1], p[:, 0, 2]
        v_lfo = tf.stack([
            inv(tf.math.log(tf.maximum(rate, 1e-3) / 0.5) / np.log(20.0)),
            tf.cos(2 * np.pi * ph), tf.sin(2 * np.pi * ph), inv(mo)], -1)
        return v_env, v_lfo

    def free_to_params(v_env, v_lfo, w):
        ue = tf.sigmoid(v_env)
        a = tf.maximum(ue[:, 0] * T_OFF, 1e-3)
        h = ue[:, 1] * (T_OFF - a)
        d = tf.maximum(ue[:, 2] * (T_OFF - a - h), 1e-3)
        p_env = tf.stack([a, h, d, ue[:, 3], ue[:, 4] * R_MAX + 1e-3], -1)
        rate = 0.5 * tf.pow(20.0, tf.sigmoid(v_lfo[:, 0]))
        phase = tf.math.floormod(
            tf.atan2(v_lfo[:, 2], v_lfo[:, 1]) / (2 * np.pi), 1.0)
        z = tf.zeros_like(rate)
        p_lfo = tf.stack([rate, phase, tf.sigmoid(v_lfo[:, 3]), z, z], -1)
        ww = w[:, None]
        return (ww * p_env + (1 - ww) * p_lfo)[:, None, :]

    def test_time_opt(p0, st, target_traj, iters, lr):
        """Stage2 の出力を初期値に、軌跡一致で勾配降下（推論時最適化）。
        非凸な逆問題を「学習された初期値 + 局所探索」で解く構成。
        モデル重みは動かさず、この入力に対するパラメータのみ最適化する。"""
        v_env, v_lfo = params_to_free(p0, st)
        v_env = tf.Variable(v_env)
        v_lfo = tf.Variable(v_lfo)
        w = st[:, 0]
        opt_t = tf.keras.optimizers.Adam(lr)

        @tf.function
        def one():
            with tf.GradientTape() as tp:
                pp = free_to_params(v_env, v_lfo, w)
                loss = tf.reduce_mean(tf.abs(reconstruct(pp, st) - target_traj))
            g = tp.gradient(loss, [v_env, v_lfo])
            opt_t.apply_gradients(zip(g, [v_env, v_lfo]))
            return loss

        for _ in range(iters):
            one()
        return free_to_params(v_env, v_lfo, w)

    @tf.function(reduce_retracing=True)
    def step(x, f0, y, t_true, p_true):
        with tf.GradientTape() as tp:
            traj = s1(x, f0)
            loss = tf.reduce_mean(tf.abs(traj - y))          # Stage1 教師（信号空間）
            if s2 is not None:
                if args.s2_input == "true":
                    inp = y                                   # 真の軌跡（上限診断）
                elif args.s2_input == "noisy":
                    inp = tf.clip_by_value(
                        y + tf.random.normal(tf.shape(y), 0.0, args.noise), 0.0, 1.0)
                else:
                    inp = tf.stop_gradient(traj) if args.detach else traj
                p, st, tl = s2(inp)
                rec = reconstruct(p, st)
                # Stage2 教師: 真の軌跡への一致（＋型 CE）
                loss += tf.reduce_mean(tf.abs(rec - y))
                loss += 0.3 * tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.cast(1.0 - t_true, tf.int32), logits=tl))
                if args.param_sup > 0:      # 第一: パラメータ空間の直接教師
                    pp = p[:, 0]                                   # [B,5]
                    is_env = t_true[:, None]
                    sc = tf.constant([1.0, 1.0, 1.0, 1.0, R_MAX])
                    d_env = tf.reduce_sum(
                        (((pp - p_true) / sc) ** 2) * is_env, -1)
                    # LFO: rate は対数、位相は円環距離
                    lr = (tf.math.log(tf.maximum(pp[:, 0], 1e-3) / 0.5)
                          - tf.math.log(tf.maximum(p_true[:, 0], 1e-3) / 0.5)
                          ) / np.log(20.0)
                    dphi = tf.abs(pp[:, 1] - p_true[:, 1])
                    circ = tf.minimum(dphi, 1.0 - dphi)
                    d_lfo = (lr ** 2 + circ ** 2
                             + (pp[:, 2] - p_true[:, 2]) ** 2) * (1.0 - t_true)
                    loss += args.param_sup * tf.reduce_mean(d_env + d_lfo)
        vs = s1.trainable_variables + (s2.trainable_variables if s2 else [])
        g = tp.gradient(loss, vs)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, vs))
        return loss

    # --- param モード: Stage1 を経由せず直接パラメータ（比較用）---
    if args.mode == "param":
        s2p = Stage2(T_OFF, args.refine, args.use_feat, SR, args.hop)
        enc = Stage1(SR, args.hop, args.mel)

        @tf.function(reduce_retracing=True)
        def step_p(x, f0, y, t_true):
            with tf.GradientTape() as tp:
                traj = enc(x, f0)                    # 中間軌跡（教師なし）
                p, st, tl = s2p(traj)
                rec = reconstruct(p, st)
                loss = tf.reduce_mean(tf.abs(rec - y))     # 教師は最終出力のみ
                loss += 0.3 * tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.cast(1.0 - t_true, tf.int32), logits=tl))
            vs = enc.trainable_variables + s2p.trainable_variables
            g = tp.gradient(loss, vs)
            g, _ = tf.clip_by_global_norm(g, 1.0)
            opt.apply_gradients(zip(g, vs))
            return loss

    print(f"mode={args.mode}  target={tname}  n_mod={args.n_mod}  "
          f"detach={args.detach}  param_sup={args.param_sup}  "
          f"refine={args.refine}  use_feat={args.use_feat}  "
          f"s2_input={args.s2_input}  frames={n_frames}")
    hist = []
    for it in range(args.steps + 1):
        x, f0, y, tt, pt = batch()
        l = step_p(x, f0, y, tt) if args.mode == "param" else step(x, f0, y, tt, pt)
        hist.append(float(l))
        if it % 500 == 0:
            print(f"  step {it:5d}  loss = {np.mean(hist[-100:]):.4f}", flush=True)

    # --- 評価 ---
    ok_s1, ok_final, mae_s1, mae_final, tacc = [], [], [], [], []
    ok_tto, mae_tto = [], []
    for _ in range(12):
        x, f0, y, tt, pt = batch()
        yn = y.numpy()
        if args.mode == "param":
            traj = enc(x, f0)
            p, st, tl = s2p(traj)
            rec = reconstruct(p, st).numpy()
        else:
            traj = s1(x, f0)
            if s2 is not None:
                if args.s2_input == "true":
                    inp_e = y
                elif args.s2_input == "noisy":
                    inp_e = tf.clip_by_value(
                        y + tf.random.normal(tf.shape(y), 0.0, args.noise), 0.0, 1.0)
                else:
                    inp_e = traj
                p, st, tl = s2(inp_e)
                rec = reconstruct(p, st).numpy()
                if args.tto > 0:
                    p2 = test_time_opt(p, st, inp_e, args.tto, args.tto_lr)
                    rec_t = reconstruct(p2, st).numpy()
            else:
                rec, tl = None, None
        t1 = np.abs(traj.numpy() - yn).mean(1)
        mae_s1 += list(t1); ok_s1 += list(t1 <= 0.1)
        if rec is not None:
            t2 = np.abs(rec - yn).mean(1)
            mae_final += list(t2); ok_final += list(t2 <= 0.1)
            tacc += list((tf.argmax(tl, -1).numpy() == (1 - tt.numpy())).astype(float))
            if args.tto > 0:
                t3 = np.abs(rec_t - yn).mean(1)
                mae_tto += list(t3); ok_tto += list(t3 <= 0.1)

    print(f"\n== mode={args.mode} target={tname} n_mod={args.n_mod} "
          f"detach={args.detach} ==")
    print(f"  Stage1（軌跡）  shape_ok = {np.mean(ok_s1):.3f}  MAE = {np.mean(mae_s1):.4f}")
    if ok_final:
        print(f"  最終（パラメータ経由） shape_ok = {np.mean(ok_final):.3f}  "
              f"MAE = {np.mean(mae_final):.4f}")
        print(f"  type_acc = {np.mean(tacc):.3f}")
        print(f"  段間の損失 = {np.mean(ok_s1) - np.mean(ok_final):+.3f}")
        if ok_tto:
            print(f"  ★ 推論時最適化後（{args.tto} 反復） shape_ok = {np.mean(ok_tto):.3f}  "
                  f"MAE = {np.mean(mae_tto):.4f}")
            print(f"     改善 = {np.mean(ok_tto) - np.mean(ok_final):+.3f}  "
                  f"（Stage1 {np.mean(ok_s1):.3f} に対する残差 "
                  f"{np.mean(ok_s1) - np.mean(ok_tto):+.3f}）")
    print("\n  参照値: free 0.729 / 単段 param 0.099 / 二段 baseline 0.359")
    if args.s2_input == "true":
        print("  読み: これは Stage2 の**上限**（入力が完璧な場合）。")
        print("        高い（>0.8）→ Stage1 の誤差が Stage2 を壊している")
        print("        低い（<0.5）→ Stage2 自体の学習能力が限界")
        print("        ceiling（勾配降下 500 反復）= 1.000 との差が「一発推定の代償」")


if __name__ == "__main__":
    main()
