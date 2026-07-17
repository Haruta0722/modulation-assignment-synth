"""
train.py — 本学習スクリプト（仕様 v7 §5 学習・§9.2 評価）
================================================================================
使い方:
  python train.py --data dataset_v7 --steps 30000 --batch 32          # 学習（自動再開）
  python train.py --data dataset_v7 --eval_only --split test          # §9.2 評価のみ

実装内容:
  - memmap データパイプライン（manifest の splits に従う）
  - Gumbel 温度アニーリング 1.0 → 0.3（指数、§5）
  - Adam + 勾配クリッピング（ADHSR の 1/a 項対策）
  - チェックポイント保存・自動再開、CSV ログ
  - 検証ループ: 損失内訳 + §9.2 指標
      2. アサイン復元 P/R/F1（マッチング後。FP=余計な配線 を個別報告）
      3. 深さ回帰誤差（真アサイン成分）
      4. ソース形状誤差（信号空間 e_k(n)、使用行のみ）
      5. 静的パラメータ誤差
      6. 音響距離 MRS
      8. 型正解率（低レート LFO 層 f·T<1 を層別）
      9. クラスタ層別（コア8 / クラスタ / 弱効き別の F1、クラスタ内誤帰属数）
注:
  - λI 版 PNP（M̄(θ) 事前計算後は estimator.losses の param 項を二次形式に差し替え）
  - OOM 時は --batch を下げる（§9.4: scan の BPTT がメモリ律速）
"""
import argparse, csv, json, os, time
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import estimator as E

K, M = ds.K_ENVS, ds.M_DEST
CORE8 = [0, 1, 2, 3, 4, 7, 8, 10]
CLUSTER = [10, 11, 15, 16]          # thresh, ratio, drive_d, mix_d（thresh は二重役割）
WEAK = [5, 6, 9, 12, 13, 14]
LAM = dict(spec=1.0, param=0.2, assign=1.0, sparse=0.03)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v7")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--temp_hi", type=float, default=1.0)
    ap.add_argument("--temp_lo", type=float, default=0.3)
    ap.add_argument("--ckpt_dir", type=str, default="ckpt_v7")
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------- データ
class Data:
    def __init__(self, root):
        self.man = json.load(open(os.path.join(root, "manifest.json")))
        n, N = self.man["n_total"], self.man["N"]
        self.wav = np.memmap(os.path.join(root, "waveforms.int16.bin"),
                             dtype=np.int16, mode="r", shape=(n, N))
        self.P = dict(np.load(os.path.join(root, "params.npz")))
        self.scale = self.man["int16_scale"]
        self.splits = {k: np.arange(v[0], v[1]) for k, v in self.man["splits"].items()}
        # --- v8.2: 正準ソース順（学習教師の安定化）。ENV 先頭 attack 昇順 → LFO rate 昇順 ---
        styp, sp = self.P["src_type"], self.P["src_params"]
        key = np.where(styp == 1.0, sp[..., 0], 100.0 + sp[..., 0])
        perm = np.argsort(key, axis=1, kind="stable")             # [n,K]
        ar = np.arange(len(key))[:, None]
        for k_ in ("src_params", "src_type", "R"):
            self.P[k_] = self.P[k_][ar, perm]
        inv = np.argsort(perm, axis=1).astype(np.int64)           # 旧→新
        z = self.P["z"].astype(np.int64)
        self.P["z"] = np.where(
            z >= 0, np.take_along_axis(inv, np.maximum(z, 0), axis=1), -1
        ).astype(np.int8)

    def batch(self, idx):
        x = tf.constant(self.wav[idx].astype(np.float32) / self.scale)
        b = {}
        for k in ("theta_stat", "theta_osc", "gmk", "src_params",
                  "src_type", "R", "f0"):
            b[k] = tf.constant(self.P[k][idx].astype(np.float32))
        b["z"] = tf.constant(self.P["z"][idx].astype(np.int32))
        return x, b


# ---------------------------------------------------------------- §9.2 指標
def metrics_np(pred, batch, mapping, sr, t_off, N):
    """マッチング後の §9.2 指標（numpy 集計）"""
    inv = tf.argsort(mapping, axis=1)
    z = batch["z"].numpy()
    z_re = np.where(z >= 0,
                    tf.gather(inv, tf.maximum(batch["z"], 0), axis=1,
                              batch_dims=1).numpy(), K)
    g_pred = tf.argmax(pred["gate_logits"], -1, output_type=tf.int32).numpy()
    B = z.shape[0]

    pred_on = g_pred < K
    true_on = z_re < K
    tp = (pred_on & true_on & (g_pred == z_re))
    fp = pred_on & ~tp                      # 余計/誤配線（誤帰属は FP かつ FN）
    fn = true_on & ~tp

    def prf(sel):
        TP, FP, FN = tp[:, sel].sum(), fp[:, sel].sum(), fn[:, sel].sum()
        p = TP / max(TP + FP, 1)
        r = TP / max(TP + FN, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        return p, r, f

    out = {}
    out["P"], out["R"], out["F1"] = prf(slice(None))
    out["FP_rate"] = fp.sum() / max(B * M, 1)
    for name, cols in (("core8", CORE8), ("cluster", CLUSTER), ("weak", WEAK)):
        out[f"F1_{name}"] = prf(cols)[2]

    # クラスタ内誤帰属: 真がクラスタ列 m の source k を、pred が別のクラスタ列 m' に配線
    R_hat = E.gather_rows(pred["R"], mapping).numpy()
    mis = 0
    for b in range(B):
        for m in CLUSTER:
            k = z_re[b, m]
            if k < K and g_pred[b, m] != k:
                for m2 in CLUSTER:
                    if m2 != m and g_pred[b, m2] == k:
                        mis += 1
                        break
    out["cluster_misattr"] = mis / max(B, 1)

    # 深さ回帰誤差（真アサイン成分の R̂ 値）
    errs = []
    Rt = batch["R"].numpy()
    for b in range(B):
        for m in range(M):
            k = z_re[b, m]
            if k < K:
                errs.append(abs(R_hat[b, k, m] - Rt[b, :, m].sum()))
    out["depth_mae"] = float(np.mean(errs)) if errs else np.nan

    # ソース形状誤差（信号空間、使用行のみ）
    used = (z_re[:, :, None] == np.arange(K)[None, None, :]).any(axis=1)  # [B,K]
    e_true = ds.sources(batch["src_params"], batch["src_type"],
                        t_off, N, sr).numpy()
    sp = E.gather_rows(pred["src_params"], mapping)
    st = E.gather_rows(pred["type_oh"][..., 0], mapping)
    e_pred = ds.sources(sp, st, t_off, N, sr).numpy()
    diff = np.abs(e_true - e_pred).mean(-1)                               # [B,K]
    out["src_mae"] = float(diff[used].mean()) if used.any() else np.nan

    # 静的パラメータ誤差
    out["stat_mae"] = float(np.abs(pred["theta_stat"].numpy()
                                   - batch["theta_stat"].numpy()).mean())
    out["gmk_mae"] = float(np.abs(pred["gmk"].numpy() - batch["gmk"].numpy()).mean())

    # 型正解率（低レート層別）
    t_lab = (1.0 - batch["src_type"].numpy()).astype(int)
    t_pred = tf.argmax(E.gather_rows(pred["type_logits"], mapping), -1,
                       output_type=tf.int32).numpy()
    out["type_acc"] = float((t_pred == t_lab).mean())
    lo = (t_lab == 1) & (batch["src_params"].numpy()[..., 0] * (N / sr) < 1.0)
    if lo.any():
        out["type_acc_lowrate"] = float((t_pred == t_lab)[lo].mean())
    return out


# ---------------------------------------------------------------- メイン
def main():
    args = parse_args()
    np.random.seed(args.seed); tf.random.set_seed(args.seed)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    data = Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    fmax = E.F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))

    model = E.Estimator(sr=SR, t_off=T_OFF)
    opt = tf.keras.optimizers.Adam(args.lr)
    step_var = tf.Variable(0, dtype=tf.int64)
    ckpt = tf.train.Checkpoint(model=model, opt=opt, step=step_var)
    mgr = tf.train.CheckpointManager(ckpt, args.ckpt_dir, max_to_keep=3)
    if mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint)
        print(f"resumed from {mgr.latest_checkpoint} (step {int(step_var)})")

    def temp_at(step):
        frac = min(step / max(args.steps, 1), 1.0)
        return float(args.temp_hi * (args.temp_lo / args.temp_hi) ** frac)

    @tf.function
    def train_step(x, batch, temp):
        with tf.GradientTape() as tape:
            pred = model(x, batch["f0"], temp=temp, hard=True)
            total, parts, _ = E.losses(pred, batch, x, SR, T_OFF, NH, LAM, matching=False)
        grads = tape.gradient(total, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, args.clip)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return total, parts

    def evaluate(split, n_eval):
        idx_all = data.splits[split][:n_eval]
        agg, losses_sum, nb = None, None, 0
        for s0 in range(0, len(idx_all), args.batch):
            idx = idx_all[s0:s0 + args.batch]
            if len(idx) < args.batch:
                idx = np.concatenate([idx, idx_all[:args.batch - len(idx)]])
            x, batch = data.batch(idx)
            pred = model(x, batch["f0"], temp=args.temp_lo, hard=True)
            total, parts, mapping = E.losses(pred, batch, x, SR, T_OFF, NH, LAM)
            m = metrics_np(pred, batch, mapping, SR, T_OFF, N)
            m["loss"] = float(total)
            m["spec"] = float(parts["spec"])
            agg = {k: agg.get(k, 0) + v for k, v in m.items()} if agg else dict(m)
            nb += 1
        return {k: v / nb for k, v in agg.items()}

    if args.eval_only:
        r = evaluate(args.split, len(data.splits[args.split]))
        print(f"== §9.2 evaluation on {args.split} ==")
        for k, v in sorted(r.items()):
            print(f"  {k:18s} = {v:.4f}")
        return

    log_path = os.path.join(args.ckpt_dir, "log.csv")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["step", "temp", "total", "spec", "param", "gate",
                       "type", "sparse", "val_F1", "val_type_acc", "val_spec"])

    rng = np.random.default_rng(args.seed + int(step_var))
    train_idx = data.splits["train"]
    t0, last = time.time(), int(step_var)
    while int(step_var) < args.steps:
        step = int(step_var)
        idx = rng.choice(train_idx, args.batch, replace=False)
        x, batch = data.batch(idx)
        temp = temp_at(step)
        total, parts = train_step(x, batch, tf.constant(temp, tf.float32))
        step_var.assign_add(1)

        if step % 50 == 0:
            rate = (int(step_var) - last) / max(time.time() - t0, 1e-9)
            t0, last = time.time(), int(step_var)
            print(f"step {step:6d} T={temp:.2f} total={float(total):8.3f} "
                  + " ".join(f"{k}={float(v):.3f}" for k, v in parts.items())
                  + f"  ({rate:.2f} it/s)", flush=True)

        val = {}
        if step > 0 and step % args.val_every == 0:
            val = evaluate("val", args.eval_n)
            print(f"  [val] F1={val['F1']:.3f} (core8 {val['F1_core8']:.3f} / "
                  f"cluster {val['F1_cluster']:.3f} / weak {val['F1_weak']:.3f}) "
                  f"FPrate={val['FP_rate']:.3f} depthMAE={val['depth_mae']:.3f} "
                  f"srcMAE={val['src_mae']:.3f} typeAcc={val['type_acc']:.3f} "
                  f"spec={val['spec']:.3f}", flush=True)
        if step > 0 and step % args.ckpt_every == 0:
            mgr.save()
        logw.writerow([step, f"{temp:.3f}", f"{float(total):.4f}"]
                      + [f"{float(parts[k]):.4f}" for k in
                         ("spec", "param", "gate", "type", "sparse")]
                      + [f"{val.get('F1', ''):}", f"{val.get('type_acc', ''):}",
                         f"{val.get('spec', ''):}"])
        logf.flush()
    mgr.save()
    print("training complete")


if __name__ == "__main__":
    main()
