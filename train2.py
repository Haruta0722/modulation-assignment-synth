"""
train2.py — 二段構成の推定器を本タスクで学習・評価する
================================================================================
train.py と同じデータ・同じ評価指標を用いるため、
単段のベースライン（F1 = 0.267 / desc_rec = 0.004）と直接比較できる。

差分:
  - 推定器を TwoStageEstimator に差し替え
  - 損失に Stage 1 の軌跡教師を追加（--traj_w で重み調整）
  - 評価時に Stage 1 の軌跡精度（traj_shape_ok）も報告

使い方:
  python train2.py --data dataset_v7 --steps 30000 --batch 32 --ckpt_dir ckpt_two
  python train2.py --data dataset_v7 --eval_only --split test --ckpt_dir ckpt_two

比較対象（単段）:
  python train.py --data dataset_v7 --eval_only --split test --ckpt_dir ckpt_nw0.25
"""
import argparse, csv, json, os, time
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import estimator as E
import estimator2 as E2
import train as T1          # Data クラスと metrics_np を再利用
import weights_io as W

K, M = ds.K_ENVS, ds.M_DEST
LAM = dict(spec=1.0, param=0.2, assign=3.0, sparse=0.0,
           none_w=0.25, traj=1.0)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v7")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--temp_hi", type=float, default=1.0)
    ap.add_argument("--temp_lo", type=float, default=0.3)
    ap.add_argument("--traj_w", type=float, default=1.0,
                    help="Stage 1 の軌跡教師の重み")
    ap.add_argument("--none_w", type=float, default=0.25)
    ap.add_argument("--spec_w", type=float, default=0.0,
                    help="再合成損失の重み（既定 0＝無効）")
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--ckpt_dir", type=str, default="ckpt_two")
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed); tf.random.set_seed(args.seed)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    LAM["traj"] = args.traj_w
    LAM["none_w"] = args.none_w
    LAM["spec"] = args.spec_w
    print(f"LAM = {LAM}")

    data = T1.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    fmax = E.F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    n_frames = 1 + (N - 1024) // args.hop
    fidx = np.minimum(np.arange(n_frames) * args.hop + 512, N - 1)

    model = E2.TwoStageEstimator(sr=SR, t_off=T_OFF, hop=args.hop)
    model(tf.zeros([1, N]), tf.constant([220.0], tf.float32), temp=1.0, hard=True)
    opt = tf.keras.optimizers.Adam(args.lr)
    step_var = tf.Variable(0, dtype=tf.int64)
    # 重みの正本は weights_io（.weights.h5）。tf.train.Checkpoint は opt と step の再開用
    ckpt = tf.train.Checkpoint(opt=opt, step=step_var)
    mgr = tf.train.CheckpointManager(ckpt, args.ckpt_dir, max_to_keep=3)
    best_dir = os.path.join(args.ckpt_dir, "best")
    best_mgr = tf.train.CheckpointManager(ckpt, best_dir, max_to_keep=1)
    best_score = -1.0

    def save_to(m):
        W.save(model, W.ckpt_weights_path(m.save()))
        W.prune(m)

    if args.eval_only:
        W.load(model, W.latest(args.ckpt_dir))
    elif mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint).expect_partial()
        W.load(model, W.ckpt_weights_path(mgr.latest_checkpoint))
        print(f"resumed from {mgr.latest_checkpoint} (step {int(step_var)})")

    def temp_at(step):
        frac = min(step / max(args.steps, 1), 1.0)
        return float(args.temp_hi * (args.temp_lo / args.temp_hi) ** frac)

    @tf.function(reduce_retracing=True)
    def train_step(x, batch, temp):
        with tf.GradientTape() as tape:
            pred = model(x, batch["f0"], temp=temp, hard=True)
            total, parts, _ = E2.losses2(pred, batch, x, SR, T_OFF, NH,
                                         LAM, args.hop)
        grads = tape.gradient(total, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, args.clip)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return total, parts

    def traj_accuracy(pred, batch):
        """Stage 1 の軌跡精度（使用中ソースのみ、shape_ok 基準）"""
        e_true = ds.sources(batch["src_params"], batch["src_type"],
                            T_OFF, N, SR).numpy()[:, :, fidx]
        tr = pred["traj"].numpy()
        z = batch["z"].numpy()
        used = np.stack([(z == k).any(axis=1) for k in range(K)], 1)
        err = np.abs(tr - e_true).mean(-1)
        return float((err[used] <= 0.1).mean()) if used.any() else np.nan

    def evaluate(split, n_eval):
        idx_all = data.splits[split][:n_eval]
        agg, nb = None, 0
        for s0 in range(0, len(idx_all), args.batch):
            idx = idx_all[s0:s0 + args.batch]
            x, batch = data.batch(idx)
            pred = model(x, batch["f0"], temp=args.temp_lo, hard=True,
                         noise=False)   # 評価は決定的に
            total, parts, mapping = E2.losses2(pred, batch, x, SR, T_OFF, NH,
                                               LAM, args.hop)
            m = T1.metrics_np(pred, batch, mapping, SR, T_OFF, N)
            m["loss"] = float(total)
            m["spec"] = float(parts["spec"])
            m["traj_loss"] = float(parts["traj"])
            m["traj_shape_ok"] = traj_accuracy(pred, batch)
            agg = {k: agg.get(k, 0) + v for k, v in m.items()} if agg else dict(m)
            nb += 1
        return {k: v / nb for k, v in agg.items()}

    if args.eval_only:
        r = evaluate(args.split, len(data.splits[args.split]))
        print(f"== two-stage evaluation on {args.split} ==")
        for k, v in sorted(r.items()):
            print(f"  {k:18s} = {v:.4f}")
        print("\n  単段ベースライン: F1 0.267 / desc_rec 0.004 / "
              "F1_core8 0.362 / F1_cluster 0.000")
        return

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.ckpt_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["step", "temp", "total", "traj", "param", "gate",
                       "type", "val_F1", "val_desc", "val_traj_ok"])

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
            print(f"step {step:6d} T={temp:.2f} total={float(total):7.3f} "
                  + " ".join(f"{k}={float(v):.3f}" for k, v in parts.items())
                  + f"  ({rate:.2f} it/s)", flush=True)

        val = {}
        if step > 0 and step % args.val_every == 0:
            val = evaluate("val", args.eval_n)
            print(f"  [val] F1={val['F1']:.3f} desc={val['desc_rec_rate']:.3f} "
                  f"traj_ok={val['traj_shape_ok']:.3f} "
                  f"det={val['F1_detect']:.3f} "
                  f"(core8 {val['F1_core8']:.3f} / cluster {val['F1_cluster']:.3f}) "
                  f"shape_ok={val['shape_ok_on_hit']:.3f} "
                  f"depth_ok={val['depth_ok_on_hit']:.3f}", flush=True)
        if val:                                  # 検証したステップで最良を更新
            score = val["F1"] + val["desc_rec_rate"]
            if score > best_score:
                best_score = score
                save_to(best_mgr)
                print(f"    best updated: F1={val['F1']:.3f} "
                      f"desc={val['desc_rec_rate']:.4f} → {args.ckpt_dir}/best",
                      flush=True)
        if step > 0 and step % args.ckpt_every == 0:
            save_to(mgr)
        logw.writerow([step, f"{temp:.3f}", f"{float(total):.4f}"]
                      + [f"{float(parts[k]):.4f}" for k in
                         ("traj", "param", "gate", "type")]
                      + [f"{val.get('F1', ''):}", f"{val.get('desc_rec_rate', ''):}",
                         f"{val.get('traj_shape_ok', ''):}"])
        logf.flush()
    save_to(mgr)
    print("training complete")


if __name__ == "__main__":
    main()
