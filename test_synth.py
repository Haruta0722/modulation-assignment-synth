"""
test_synth.py — 微分可能シンセの動作検証
  [1] 仕様準拠の乱数パラメータで順方向 → 有限性・振幅の確認
  [2] 全パラメータ群への勾配流の確認（NaN/None が無いこと）
  [3] アサイン固定でのサウンドマッチング縮小実験（工程3のミニ版）:
      真値から摂動した連続パラメータを Adam で最適化し、損失が下がることを確認
注: 検証用にクリップ長・t_off は縮小（実データ仕様は T=1.5s, t_off=1.0s, N=24000）
"""
import numpy as np
import tensorflow as tf
import diff_synth_tf as ds

tf.random.set_seed(0)
np.random.seed(0)

SR = 16000
T_CLIP = 0.10                 # 検証用に短縮
N = int(SR * T_CLIP)
T_OFF = 0.06
B, K, M = 2, ds.K_ENVS, ds.M_DEST
F0 = np.array([220.0, 220.0], np.float32)
FMAX = float(F0.max() * 2 ** (100 / 1200))
H_SAW = int((SR / 2) // FMAX)
H_SQR = int(((SR / 2) / FMAX - 1) // 2)

# ---------------------------------------------------------------- [0] 仕様準拠サンプリング
def sample_params(rng):
    theta = rng.uniform(0.1, 0.6, size=(B, M)).astype(np.float32)
    theta[:, 16] = rng.uniform(0.1, 0.9, size=B)                    # mix ≥ 0.1
    # 正準順序 fin: θ3+0.05 ≤ θ4 (0-based 2,3)
    lo = np.minimum(theta[:, 2], theta[:, 3])
    theta[:, 2] = lo
    theta[:, 3] = np.maximum(lo + 0.05, np.maximum(theta[:, 2], theta[:, 3]))
    R = np.zeros((B, K, M), np.float32)
    z_true = np.full((B, M), -1, np.int64)                          # -1 = ∅
    for b in range(B):
        cols = [rng.choice([0, 1])]                                 # D4=(b): level に必ず1本
        others = rng.choice([m for m in range(M) if m not in cols], size=3, replace=False)
        cols += list(others)
        for m in cols:
            k = rng.integers(0, K)
            depth_max = min(1.0 - theta[b, m], 1.0)
            depth = rng.uniform(0.2, max(0.21, depth_max))
            sign = 1.0 if (m in (0, 1) or rng.random() < 0.5) else -1.0  # level は正で確実に発音
            R[b, k, m] = sign * depth
            z_true[b, m] = k
    # ADHSR: 割合パラメータ化で a+h+d ≤ t_off を担保
    u = rng.uniform(0.05, 0.6, size=(B, K, 3)).astype(np.float32)
    a = u[..., 0] * T_OFF
    h = u[..., 1] * (T_OFF - a)
    d = np.maximum(u[..., 2] * (T_OFF - a - h), 1e-3)
    s = rng.uniform(0.2, 1.0, size=(B, K)).astype(np.float32)
    r = rng.uniform(0.005, 0.03, size=(B, K)).astype(np.float32)
    env = np.stack([a, h, d, s, r], axis=-1).astype(np.float32)
    osc = rng.uniform(0.0, 1.0, size=(B, 2)).astype(np.float32)
    gmk = rng.uniform(0.0, 0.5, size=(B,)).astype(np.float32)
    return theta, osc, gmk, env, R, z_true

rng = np.random.default_rng(42)
theta_t, osc_t, gmk_t, env_t, R_t, z_true = sample_params(rng)

@tf.function
def forward(theta, osc, gmk, env, R):
    return ds.synth(theta, osc, gmk, env, R,
                    tf.constant(F0), T_OFF, N, SR, H_SAW, H_SQR)

# ---------------------------------------------------------------- [1] 順方向
x_true = forward(*[tf.constant(v) for v in (theta_t, osc_t, gmk_t, env_t, R_t)])
x_np = x_true.numpy()
print(f"[1] forward OK  shape={x_np.shape}  finite={np.isfinite(x_np).all()}  "
      f"rms={np.sqrt((x_np**2).mean()):.4f}  peak={np.abs(x_np).max():.4f}")
assert np.isfinite(x_np).all()

# ---------------------------------------------------------------- [2] 勾配流
vars_ = {n: tf.Variable(v) for n, v in
         [("theta", theta_t), ("osc", osc_t), ("gmk", gmk_t), ("env", env_t), ("R", R_t)]}
target = tf.constant(x_np * 0.9)  # 少しずらした擬似ターゲット
with tf.GradientTape() as tape:
    xp = forward(vars_["theta"], vars_["osc"], vars_["gmk"], vars_["env"], vars_["R"])
    loss = tf.reduce_sum(ds.mrstft_loss(xp, target, fft_sizes=(128, 256, 512)))
grads = tape.gradient(loss, list(vars_.values()))
print("[2] gradient flow:")
ok = True
for (name, _), g in zip(vars_.items(), grads):
    if g is None:
        print(f"    {name:6s}: None  <-- NG"); ok = False; continue
    gn = g.numpy()
    frac = float((np.abs(gn) > 0).mean())
    print(f"    {name:6s}: finite={np.isfinite(gn).all()}  |g|max={np.abs(gn).max():.3e}  nonzero={frac:.0%}")
    ok = ok and np.isfinite(gn).all()
assert ok, "gradient check failed"

# ---------------------------------------------------------------- [3] アサイン固定ミニフィット
gate_onehot = np.zeros((B, K, M), np.float32)
d_true = np.zeros((B, M), np.float32)
for b in range(B):
    for m in range(M):
        if z_true[b, m] >= 0:
            gate_onehot[b, z_true[b, m], m] = 1.0
            d_true[b, m] = R_t[b, z_true[b, m], m]

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

pert = lambda v, s: (v + rng.normal(0, s, v.shape)).astype(np.float32)
v_theta = tf.Variable(logit(pert(theta_t, 0.08)))
v_osc   = tf.Variable(logit(pert(osc_t, 0.08)))
v_gmk   = tf.Variable(logit(pert(gmk_t, 0.08)))
v_uenv  = tf.Variable(pert(np.stack([env_t[..., 0] / T_OFF,
                                     env_t[..., 1] / np.maximum(T_OFF - env_t[..., 0], 1e-4),
                                     env_t[..., 2] / np.maximum(T_OFF - env_t[..., 0] - env_t[..., 1], 1e-4),
                                     env_t[..., 3],
                                     env_t[..., 4] / 0.05], axis=-1), 0.10))
v_d     = tf.Variable(np.arctanh(np.clip(pert(d_true, 0.15), -0.99, 0.99)).astype(np.float32))
gate_c  = tf.constant(gate_onehot)

def build_params():
    theta = tf.sigmoid(v_theta)
    osc   = tf.sigmoid(v_osc)
    gmk   = tf.sigmoid(v_gmk)
    u     = tf.sigmoid(v_uenv)
    a = u[..., 0] * T_OFF
    h = u[..., 1] * (T_OFF - a)
    d = tf.maximum(u[..., 2] * (T_OFF - a - h), 1e-3)
    s = u[..., 3]
    r = u[..., 4] * 0.05 + 1e-3
    env = tf.stack([a, h, d, s, r], axis=-1)
    R = gate_c * tf.tanh(v_d)[:, None, :]                       # 列スパース×深さ（§4）
    return theta, osc, gmk, env, R

opt = tf.keras.optimizers.Adam(2e-2)
tvars = [v_theta, v_osc, v_gmk, v_uenv, v_d]

@tf.function
def fit_step():
    with tf.GradientTape() as tp:
        xp = forward(*build_params())
        l = tf.reduce_sum(ds.mrstft_loss(xp, x_true, fft_sizes=(128, 256, 512)))
    gs = tp.gradient(l, tvars)
    opt.apply_gradients(zip(gs, tvars))
    return l

print("[3] assignment-fixed mini fit (工程3ミニ版):")
l0 = None
for it in range(31):
    l = fit_step().numpy()
    if l0 is None: l0 = l
    if it % 5 == 0:
        print(f"    step {it:3d}  loss = {l:.4f}")
print(f"    loss: {l0:.4f} -> {l:.4f}  ({(1 - l / l0) * 100:.1f}% reduction)")
assert l < l0, "loss did not decrease"
print("ALL CHECKS PASSED")
