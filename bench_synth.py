"""
bench_synth.py — 素朴版 vs 高速版の数値一致検証と速度ベンチ
使い方:
  python bench_synth.py check              … 一致検証（小さめN）
  python bench_synth.py bench <impl> <sr> [T]  … impl ∈ {naive,fast}, sr ∈ {16000,44100}
"""

import sys, time
import numpy as np
import tensorflow as tf
import diff_synth_tf as ds
import diff_synth_fast as fast

tf.random.set_seed(0)
np.random.seed(0)

B, K, M = 2, ds.K_ENVS, ds.M_DEST


def sample_params(rng, t_off):
    theta = rng.uniform(0.1, 0.6, size=(B, M)).astype(np.float32)
    theta[:, 16] = rng.uniform(0.1, 0.9, size=B)
    lo = np.minimum(theta[:, 2], theta[:, 3])
    theta[:, 2] = lo
    theta[:, 3] = np.maximum(lo + 0.05, np.maximum(theta[:, 2], theta[:, 3]))
    R = np.zeros((B, K, M), np.float32)
    for b in range(B):
        cols = [rng.choice([0, 1])]
        cols += list(
            rng.choice(
                [m for m in range(M) if m not in cols], size=3, replace=False
            )
        )
        for m in cols:
            k = rng.integers(0, K)
            depth = rng.uniform(0.2, max(0.21, min(1.0 - theta[b, m], 1.0)))
            sign = 1.0 if (m in (0, 1) or rng.random() < 0.5) else -1.0
            R[b, k, m] = sign * depth
    u = rng.uniform(0.05, 0.6, size=(B, K, 3)).astype(np.float32)
    a = u[..., 0] * t_off
    h = u[..., 1] * (t_off - a)
    d = np.maximum(u[..., 2] * (t_off - a - h), 1e-3)
    s = rng.uniform(0.2, 1.0, size=(B, K)).astype(np.float32)
    r = rng.uniform(0.005, 0.05, size=(B, K)).astype(np.float32)
    env = np.stack([a, h, d, s, r], axis=-1).astype(np.float32)
    osc = rng.uniform(0.0, 1.0, size=(B, 2)).astype(np.float32)
    gmk = rng.uniform(0.0, 0.5, size=(B,)).astype(np.float32)
    return theta, osc, gmk, env, R


def harmonics(sr, f0max):
    fmax = f0max * 2 ** (100 / 1200)
    return int((sr / 2) // fmax), int(((sr / 2) / fmax - 1) // 2)


# ---------------------------------------------------------------- 一致検証
def run_check():
    SR, T, T_OFF = 16000, 0.2, 0.12
    N = int(SR * T)
    F0 = np.array([220.0, 220.0], np.float32)
    Hs, Hq = harmonics(SR, F0.max())
    rng = np.random.default_rng(7)
    theta, osc, gmk, env, R = sample_params(rng, T_OFF)
    tc = [tf.constant(v) for v in (theta, osc, gmk, env, R)]
    f0c = tf.constant(F0)

    @tf.function
    def naive_stages():
        envs = ds.adhsr(tc[3], T_OFF, N, SR)
        p = ds.modulate(tc[0], tc[4], envs, SR)
        x0 = ds.oscillator(p, tc[1], f0c, N, SR, Hs, Hq)
        x1 = ds.svf_filter(x0, p["cutoff"], p["drive_f"], p["res_q"], SR)
        x2 = ds.chorus(x1, p["rate"], p["depth"], p["feedback"], SR)
        gdb = ds.GMK_LO + tc[2] * (ds.GMK_HI - ds.GMK_LO)
        x3 = ds.compressor(x2, p["thresh"], p["ratio"], gdb, SR)
        x4 = ds.eq3(x3, p["eq_low"], p["eq_mid"], p["eq_high"], SR)
        return x0, x1, x2, x3, x4

    @tf.function
    def fast_stages():
        envs = ds.adhsr(tc[3], T_OFF, N, SR)
        p = ds.modulate(tc[0], tc[4], envs, SR)
        x0n = ds.oscillator(p, tc[1], f0c, N, SR, Hs, Hq)  # 同一入力で段別比較
        x0f = fast.oscillator_fast(p, tc[1], f0c, N, SR, Hs, Hq)
        x1n = ds.svf_filter(x0n, p["cutoff"], p["drive_f"], p["res_q"], SR)
        x1f = fast.svf_filter_fast(
            x0n, p["cutoff"], p["drive_f"], p["res_q"], SR
        )
        x2f = fast.chorus_fast(x1n, p["rate"], p["depth"], p["feedback"], SR)
        gdb = ds.GMK_LO + tc[2] * (ds.GMK_HI - ds.GMK_LO)
        x2n = ds.chorus(x1n, p["rate"], p["depth"], p["feedback"], SR)
        x3f = fast.compressor_fast(x2n, p["thresh"], p["ratio"], gdb, SR)
        x3n = ds.compressor(x2n, p["thresh"], p["ratio"], gdb, SR)
        x4f = fast.eq3_fast(x3n, p["eq_low"], p["eq_mid"], p["eq_high"], SR)
        return x0f, x1f, x2f, x3f, x4f

    xs_n = [v.numpy() for v in naive_stages()]
    xs_f = [v.numpy() for v in fast_stages()]

    print("stage-wise max|Δ| (abs, rel):")
    for name, a, b in zip(["osc", "svf", "chorus", "comp", "eq"], xs_n, xs_f):
        d = np.abs(a - b).max()
        r = d / (np.abs(a).max() + 1e-9)
        print(f"  {name:7s}: {d:.3e}  ({r:.3e})")
        assert r < 1e-3, name

    @tf.function
    def full_naive():
        return ds.synth(
            tc[0], tc[1], tc[2], tc[3], tc[4], f0c, T_OFF, N, SR, Hs, Hq
        )

    @tf.function
    def full_fast():
        return fast.synth_fast(
            tc[0], tc[1], tc[2], tc[3], tc[4], f0c, T_OFF, N, SR, Hs, Hq
        )

    yn, yf = full_naive().numpy(), full_fast().numpy()
    d = np.abs(yn - yf).max()
    r = d / (np.abs(yn).max() + 1e-9)
    print(f"  full   : {d:.3e}  ({r:.3e})")
    assert r < 3e-3

    v = tf.Variable(R)

    @tf.function
    def grad_naive():
        with tf.GradientTape() as t1:
            l = tf.reduce_sum(
                ds.synth(
                    tc[0], tc[1], tc[2], tc[3], v, f0c, T_OFF, N, SR, Hs, Hq
                )
                ** 2
            )
        return t1.gradient(l, v)

    @tf.function
    def grad_fast():
        with tf.GradientTape() as t2:
            l = tf.reduce_sum(
                fast.synth_fast(
                    tc[0], tc[1], tc[2], tc[3], v, f0c, T_OFF, N, SR, Hs, Hq
                )
                ** 2
            )
        return t2.gradient(l, v)

    g1, g2 = grad_naive().numpy(), grad_fast().numpy()
    gr = np.abs(g1 - g2).max() / (np.abs(g1).max() + 1e-9)
    print(f"  grad(R): rel max|Δ| = {gr:.3e}")
    assert gr < 5e-3
    print("EQUIVALENCE CHECK PASSED")


# ---------------------------------------------------------------- ベンチ
def run_bench(impl, sr, T=1.5):
    SR = int(sr)
    T_OFF = 1.0 * T / 1.5
    N = int(SR * T)
    F0 = np.array([220.0] * B, np.float32)
    Hs, Hq = harmonics(SR, F0.max())
    rng = np.random.default_rng(7)
    theta, osc, gmk, env, R = sample_params(rng, T_OFF)
    fn = ds.synth if impl == "naive" else fast.synth_fast
    consts = [tf.constant(v) for v in (theta, osc, gmk, env)]
    vR = tf.Variable(R)
    f0c = tf.constant(F0)

    @tf.function
    def fwd():
        return fn(
            consts[0],
            consts[1],
            consts[2],
            consts[3],
            vR,
            f0c,
            T_OFF,
            N,
            SR,
            Hs,
            Hq,
        )

    @tf.function
    def fwd_bwd():
        with tf.GradientTape() as tp:
            x = fn(
                consts[0],
                consts[1],
                consts[2],
                consts[3],
                vR,
                f0c,
                T_OFF,
                N,
                SR,
                Hs,
                Hq,
            )
            l = tf.reduce_sum(ds.mrstft_loss(x, tf.zeros_like(x)))
        return tp.gradient(l, vR)

    t0 = time.perf_counter()
    fwd().numpy()
    trace_f = time.perf_counter() - t0
    t0 = time.perf_counter()
    fwd().numpy()
    tf_run = time.perf_counter() - t0
    t0 = time.perf_counter()
    fwd_bwd().numpy()
    trace_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    fwd_bwd().numpy()
    tb_run = time.perf_counter() - t0
    print(
        f"RESULT impl={impl} sr={SR} T={T}s B={B} N={N} "
        f"fwd={tf_run:.2f}s fwd+bwd={tb_run:.2f}s "
        f"(trace: {trace_f:.1f}/{trace_b:.1f}s)"
    )


if __name__ == "__main__":
    if sys.argv[1] == "check":
        run_check()
    else:
        run_bench(
            sys.argv[2],
            sys.argv[3],
            float(sys.argv[4]) if len(sys.argv) > 4 else 1.5,
        )
