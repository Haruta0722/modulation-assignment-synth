"""
generate_dataset.py — 仕様 v7 §9.1 準拠の合成データ生成
================================================================================
生成内容（1件あたり）:
  波形 x[N] (int16, 固定スケール) と真値 (θ_stat, θ_osc, G_mk, ソース×K(タイプ+パラメータ), R / z / depth, f0)

§9.1 の実装対応:
  - ルーティング: 総本数 J ~ U{1..J_max}(J_max=6)、列は非復元抽出（列スパース自動遵守）
    D4=(b): level1/level2 のどちらかへ必ず1本を先に割当（この1本の深さは正符号、
    かつそのソースは ENV タイプに固定＝v7。音がノート形状で鳴るという D4 の趣旨から）
  - ソースタイプ(v7): τ_k ~ Bern(1/2)。LFO は rate 対数一様[0.1,10]Hz, φ~U[0,1), m~U[0,1]
    （スロット3-4はゼロ格納・評価対象外）
  - 深さ: |R| ~ U[0.2,1]、符号一様（上記の必須 level 分のみ正）
  - 飽和回避: 各アサイン列で (θ_m, d_m) ペアを棄却サンプリング（θ_m+|d_m|≤1 まで再抽選）
  - 端領域: level1/2・mix_d は θ ≥ 0.1、chorus depth=0 と s_k=1 は含める
  - 正準順序: θ_fin1 + 0.05 ≤ θ_fin2（違反ペアは fin 2列を再抽選）
  - 固定条件: f0 = MIDI {36,48,60,72,84} 一様、t_off=1.0s, T=1.5s, sr=16kHz
  - 分割: train 90k / val 5k / test 5k（インデックス順で区切り、生成はシード固定で決定的）

保存形式（LLM パイプラインと同じ memmap 方式）:
  out/waveforms.int16.bin   [N_total, N] int16 memmap（スケールは manifest 記載）
  out/params.npz            真値一式
  out/manifest.json         定数・分割・シード・進捗
  再開可能: manifest の done カウントから続きを生成

使い方:
  python generate_dataset.py --n_total 100000 --out dataset_v6 --batch 64
  python generate_dataset.py --smoke            # 64件の動作確認＋整合検査
"""
import argparse, json, os, time
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
from diff_synth_fast import synth_fast

assert ds.LAYOUT == "mixed", "データセットは型が混在する 4 ソースで生成する（typed は読み込み時に並べ替える）"
K, M = ds.K_ENVS, ds.M_DEST
MIDI_SET = np.array([36, 48, 60, 72, 84], np.int16)
J_MAX = 6
DEPTH_LO = 0.2
THETA_LO = np.zeros(M, np.float32)
THETA_LO[[0, 1, 16]] = 0.1          # level1, level2, mix_d は端領域除外
FIN1, FIN2 = 2, 3
FIN_MARGIN = 0.05
INT16_SCALE = 4096.0                 # x_int = clip(x*4096) → ±8.0 まで表現可（クリップ回避優先）
EPS_T = 1e-3


def midi_to_f0(m):
    return 440.0 * 2.0 ** ((m.astype(np.float32) - 69.0) / 12.0)


# ---------------------------------------------------------------- パラメータ生成（全件を先に決定的に生成）
def sample_all_params(n, rng):
    # --- ルーティング（列選択） ---
    z = np.full((n, M), -1, np.int8)                     # -1 = ∅
    depth = np.zeros((n, M), np.float32)
    J = rng.integers(1, J_MAX + 1, size=n)
    lvl = rng.integers(0, 2, size=n)                     # D4: 必須 level 列
    for i in range(n):                                   # 列選択のみループ（軽い）
        cols = {int(lvl[i])}
        others = [m for m in range(M) if m not in cols]
        extra = rng.choice(others, size=int(J[i]) - 1, replace=False) if J[i] > 1 else []
        cols.update(int(c) for c in extra)
        for m in cols:
            z[i, m] = rng.integers(0, K)
    assigned = z >= 0

    # --- 深さ（符号: 必須 level 分のみ正） ---
    mag = rng.uniform(DEPTH_LO, 1.0, size=(n, M)).astype(np.float32)
    sgn = np.where(rng.random((n, M)) < 0.5, 1.0, -1.0).astype(np.float32)
    sgn[np.arange(n), lvl] = 1.0
    depth = np.where(assigned, mag * sgn, 0.0).astype(np.float32)

    # --- θ_stat: 列ごと棄却サンプリング（飽和回避 θ+|d|≤1、端領域下限） ---
    theta = rng.uniform(THETA_LO[None, :], 1.0, size=(n, M)).astype(np.float32)
    for it in range(300):
        bad = assigned & (theta + np.abs(depth) > 1.0)
        if not bad.any():
            break
        nb = int(bad.sum())
        theta[bad] = rng.uniform(np.broadcast_to(THETA_LO, (n, M))[bad], 1.0,
                                 size=nb).astype(np.float32)
        mag_new = rng.uniform(DEPTH_LO, 1.0, size=nb).astype(np.float32)
        depth[bad] = np.sign(depth[bad]) * mag_new
    else:
        raise RuntimeError("saturation rejection did not converge")

# --- fin 正準順序（構成的に保証: 棄却なし） ---
    cap1 = 1.0 - np.abs(depth[:, FIN1])          # 飽和回避の上限
    cap2 = 1.0 - np.abs(depth[:, FIN2])
    hi1 = np.minimum(cap1, cap2 - FIN_MARGIN)    # fin1 が取れる上限
    hi1 = np.maximum(hi1, 0.0)
    theta[:, FIN1] = rng.random(n).astype(np.float32) * hi1
    lo2 = theta[:, FIN1] + FIN_MARGIN            # fin2 の下限
    theta[:, FIN2] = (lo2 + rng.random(n).astype(np.float32)
                      * np.maximum(cap2 - lo2, 0.0)).astype(np.float32)
    # --- ソースタイプ（v7）: Bern(1/2)。D4 必須 level のソースは ENV に固定 ---
    styp = (rng.random((n, K)) < 0.5).astype(np.float32)      # 1=ENV, 0=LFO
    k_lvl = z[np.arange(n), lvl].astype(int)
    styp[np.arange(n), k_lvl] = 1.0

    # --- ADHSR（割合パラメータ化で a+h+d ≤ t_off を構造保証、s=1 含む） ---
    u = rng.uniform(0.0, 1.0, size=(n, K, 3)).astype(np.float32)
    return theta, depth, z, u, lvl, styp


def build_env(u, s, r, t_off):
    a = np.maximum(u[..., 0] * t_off, EPS_T)
    h = u[..., 1] * (t_off - a)
    d = np.maximum(u[..., 2] * (t_off - a - h), EPS_T)
    return np.stack([a, h, d, s, r], axis=-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_total", type=int, default=100000)
    ap.add_argument("--out", type=str, default="dataset_v6")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--toff", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true", help="64件で動作確認＋整合検査")
    ap.add_argument("--lfo_tri", type=str, default="tri", choices=["tri", "v7"],
                    help="LFO の三角波の式（実験 47）。tri = 正しい三角波、v7 = dataset_v7 の誤った式")
    args = ap.parse_args()
    if args.smoke:
        args.n_total, args.out, args.batch = 64, args.out + "_smoke", 16

    ds.set_lfo_tri(args.lfo_tri)
    SR, T, T_OFF = args.sr, args.T, args.toff
    N = int(SR * T)
    n = args.n_total
    os.makedirs(args.out, exist_ok=True)
    wav_path = os.path.join(args.out, "waveforms.int16.bin")
    par_path = os.path.join(args.out, "params.npz")
    man_path = os.path.join(args.out, "manifest.json")

    # ---------------- 真値の生成（決定的・一括） ----------------
    rng = np.random.default_rng(args.seed)
    theta, depth, z, u_env, lvl, styp = sample_all_params(n, rng)
    s_lvl = rng.uniform(0.0, 1.0, size=(n, K)).astype(np.float32)   # s=1 含む
    r_rel = rng.uniform(EPS_T, 0.6, size=(n, K)).astype(np.float32)
    env_phys = build_env(u_env, s_lvl, r_rel, T_OFF)
    # LFO 物理パラメータ（スロット共用: rate, phase, morph, 0, 0）
    rate = (0.1 * (100.0 ** rng.uniform(0.0, 1.0, size=(n, K)))).astype(np.float32)
    phase = rng.uniform(0.0, 1.0, size=(n, K)).astype(np.float32)
    morph = rng.uniform(0.0, 1.0, size=(n, K)).astype(np.float32)
    zeros = np.zeros((n, K), np.float32)
    lfo_phys = np.stack([rate, phase, morph, zeros, zeros], axis=-1)
    env = np.where(styp[..., None] == 1.0, env_phys, lfo_phys).astype(np.float32)
    osc = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
    gmk = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
    midi = MIDI_SET[rng.integers(0, len(MIDI_SET), size=n)]
    f0 = midi_to_f0(midi)
    R = np.zeros((n, K, M), np.float32)
    ii, mm = np.nonzero(z >= 0)
    R[ii, z[ii, mm].astype(int), mm] = depth[ii, mm]

    np.savez(par_path, theta_stat=theta, theta_osc=osc, gmk=gmk,
             src_params=env, src_type=styp, R=R, z=z, depth=depth, f0=f0, midi=midi)

    # ---------------- manifest（進捗込み、再開可能） ----------------
    splits = {"train": [0, max(n - 10000, 0)],
              "val":   [max(n - 10000, 0), max(n - 5000, 0)],
              "test":  [max(n - 5000, 0), n]} if n >= 100000 else \
             {"train": [0, int(n * 0.9)], "val": [int(n * 0.9), int(n * 0.95)],
              "test": [int(n * 0.95), n]}
    man = dict(spec="v7", n_total=n, sr=SR, T=T, t_off=T_OFF, N=N,
               int16_scale=INT16_SCALE, seed=args.seed, J_max=J_MAX,
               depth_lo=DEPTH_LO, midi_set=MIDI_SET.tolist(), splits=splits,
               done=0, clipped_samples=0, lfo_tri=args.lfo_tri)
    if os.path.exists(man_path):
        old = json.load(open(man_path))
        if old.get("lfo_tri", "v7") != args.lfo_tri:
            raise SystemExit(f"{man_path} は lfo_tri={old.get('lfo_tri', 'v7')} で生成途中。--lfo_tri をそろえる")
        man.update({k: v for k, v in old.items() if k in ("done", "clipped_samples")})

    wav = np.memmap(wav_path, dtype=np.int16, mode="r+" if os.path.exists(wav_path) else "w+",
                    shape=(n, N))

    # ---------------- 合成（MIDI ごとにグループ化して固定バッチ） ----------------
    B = args.batch
    fwd_cache = {}

    def get_fwd(n_harm_saw, n_harm_sqr):
        key = (n_harm_saw, n_harm_sqr)
        if key not in fwd_cache:
            @tf.function
            def fwd(t, o, g, e, Rb, f0b, st):
                return synth_fast(t, o, g, e, Rb, f0b, T_OFF, N, SR,
                                  n_harm_saw, n_harm_sqr, src_type=st)
            fwd_cache[key] = fwd
        return fwd_cache[key]

    start, t0 = man["done"], time.time()
    todo = np.arange(start, n)
    clip_count = man["clipped_samples"]
    for m_val in MIDI_SET:
        idx = todo[midi[todo] == m_val]
        if len(idx) == 0:
            continue
        fmax = float(midi_to_f0(np.array([m_val]))[0]) * 2 ** (100 / 1200)
        HS, HQ = int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2)
        fwd = get_fwd(HS, HQ)
        for s0 in range(0, len(idx), B):
            sel = idx[s0:s0 + B]
            pad = B - len(sel)
            selp = np.concatenate([sel, np.repeat(sel[-1:], pad)]) if pad else sel
            x = fwd(tf.constant(theta[selp]), tf.constant(osc[selp]),
                    tf.constant(gmk[selp]), tf.constant(env[selp]),
                    tf.constant(R[selp]), tf.constant(f0[selp]),
                    tf.constant(styp[selp])).numpy()
            x = x[:len(sel)]
            xi = np.clip(x * INT16_SCALE, -32767, 32767)
            clip_count += int((np.abs(x * INT16_SCALE) > 32767).any(axis=1).sum())
            wav[sel] = xi.astype(np.int16)
            man["done"] = man["done"] + len(sel)
            man["clipped_samples"] = clip_count
            json.dump(man, open(man_path, "w"), indent=1)
            done, dt = man["done"] - start, time.time() - t0
            if (s0 // B) % 20 == 0:
                rate = done / max(dt, 1e-9)
                eta = (n - man["done"]) / max(rate, 1e-9)
                print(f"  midi={m_val}  done={man['done']}/{n}  "
                      f"{rate:.1f} samp/s  ETA {eta/3600:.2f} h", flush=True)
    wav.flush()
    print(f"generation complete: {man['done']}/{n}, clipped={clip_count}, "
          f"total {time.time()-t0:.1f}s")

    # ---------------- 整合検査（smoke 時は必ず、通常時も軽く） ----------------
    print("integrity checks:")
    a_ok = bool(((theta + np.abs(depth) <= 1.0 + 1e-6) | (z < 0)).all())
    print(f"  saturation θ+|d|≤1 (assigned cols): {a_ok}")
    print(f"  D4 level assignment:                {bool((z[np.arange(n), lvl] >= 0).all())}")
    k_lvl = z[np.arange(n), lvl].astype(int)
    print(f"  D4 level source is ENV (v7):        {bool((styp[np.arange(n), k_lvl] == 1.0).all())}")
    n_lfo = int((styp == 0).sum())
    print(f"  source types: ENV {int((styp==1).sum())} / LFO {n_lfo} "
          f"(LFO率 {n_lfo/(n*K):.2f})")
    lfo_rates = env[..., 0][styp == 0]
    print(f"  LFO rate in [0.1,10]:               {bool(((lfo_rates>=0.1)&(lfo_rates<=10.0)).all()) if n_lfo else 'n/a'}")
    print(f"  fin canonical order:                {bool((theta[:, FIN1] + FIN_MARGIN <= theta[:, FIN2] + 1e-6).all())}")
    print(f"  column sparsity (by construction):  True")
    print(f"  depth in ±[0.2,1]:                  {bool((np.abs(depth[z >= 0]) >= DEPTH_LO - 1e-6).all())}")
    print(f"  theta lower bounds:                 {bool((theta >= THETA_LO[None, :] - 1e-6).all())}")
    if n_lfo:
        e_l = ds.lfo(tf.constant(env[:256]), N, SR).numpy()[styp[:256] == 0]
        print(f"  LFO waveform in [0,1] (lfo_tri={ds.LFO_TRI}): {bool((e_l.min() >= -1e-5) & (e_l.max() <= 1 + 1e-5))}"
              f"  (min {e_l.min():.3f}, max {e_l.max():.3f})")
    # ラウンドトリップ: 保存波形 vs 再合成（int16 量子化以内）
    i0 = 0
    fmax = float(f0[i0]) * 2 ** (100 / 1200)
    HS, HQ = int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2)
    x_re = synth_fast(tf.constant(theta[i0:i0+1]), tf.constant(osc[i0:i0+1]),
                      tf.constant(gmk[i0:i0+1]), tf.constant(env[i0:i0+1]),
                      tf.constant(R[i0:i0+1]), tf.constant(f0[i0:i0+1]),
                      T_OFF, N, SR, HS, HQ,
                      src_type=tf.constant(styp[i0:i0+1])).numpy()[0]
    stored = wav[i0].astype(np.float32) / INT16_SCALE
    err = np.abs(stored - x_re).max()
    print(f"  round-trip max|Δ| (≤ 量子化 {1/INT16_SCALE:.1e} 程度): {err:.2e}")


if __name__ == "__main__":
    main()
