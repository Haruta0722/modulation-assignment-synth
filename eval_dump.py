r"""eval_dump.py — 評価に必要な配列を .npz に保存する（指標は一切計算しない）

指標を後から何度でも定義し直せるようにするためのスクリプト。
モデルを1回流して、1サンプルごとの予測と正解を配列のまま書き出す。

使い方:
  python eval_dump.py --model two    --ckpt_dir ckpt_two      --out dump_two.npz
  python eval_dump.py --model single --ckpt_dir ckpt_nw0.25   --out dump_single.npz

保存される配列（B = テスト件数, K = 4, M = 17, T = フレーム数）:
  idx         [B]      データセット内の通し番号（2つのモデルを突き合わせる鍵）
  z_true      [B,M]    真の割り当て（0..K-1、未割り当ては -1）
  z_pred      [B,M]    ゲートの arg-max（K は未割り当て）
  d_true      [B,M]    列ごとの真の深さ（R の列和）
  d_pred      [B,M]    同じく予測
  e_true      [B,K,T]  真のソース信号（評価フレーム上）
  e_param     [B,K,T]  予測パラメータから再合成したソース信号（shape_ok が使う量）
  traj        [B,K,T]  Stage 1 の生の軌跡（single では全て 0）
  type_true   [B,K]    1 = ENV, 0 = LFO
  type_pred   [B,K]
  stat_true   [B,M]    静的値 θ_stat
  stat_pred   [B,M]
  used        [B,K]    そのソースがどこかの列で使われているか
  f0          [B]
"""
import argparse, os
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import estimator as E
import estimator2 as E2
import train as T1

K, M = ds.K_ENVS, ds.M_DEST


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v7")
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--model", type=str, required=True, choices=["single", "two"])
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--arch", type=str, default="v2", choices=["v1", "v2"],
                    help="single のときのみ使用")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--temp_lo", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=0, help="先頭 n 件だけ（0 は全件）")
    return ap.parse_args()


def main():
    args = parse_args()
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    data = T1.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    n_frames = 1 + (N - 1024) // args.hop
    fidx = np.minimum(np.arange(n_frames) * args.hop + 512, N - 1)

    if args.model == "two":
        model = E2.TwoStageEstimator(sr=SR, t_off=T_OFF, hop=args.hop)
    else:
        model = E.Estimator(sr=SR, t_off=T_OFF, arch=args.arch)
    model(tf.zeros([1, N]), tf.constant([220.0], tf.float32), temp=1.0, hard=True)

    ckpt = tf.train.Checkpoint(model=model)
    best = os.path.join(args.ckpt_dir, "best")
    path = tf.train.latest_checkpoint(best) or tf.train.latest_checkpoint(args.ckpt_dir)
    if path is None:
        raise SystemExit(f"チェックポイントが見つかりません: {args.ckpt_dir}")
    ckpt.restore(path).expect_partial()
    print(f"restored {path}")

    idx_all = data.splits[args.split]
    if args.n:
        idx_all = idx_all[:args.n]
    acc = {k: [] for k in ("idx", "z_true", "z_pred", "d_true", "d_pred",
                           "e_true", "e_param", "traj", "type_true", "type_pred",
                           "stat_true", "stat_pred", "used", "f0")}

    for s0 in range(0, len(idx_all), args.batch):
        idx = idx_all[s0:s0 + args.batch]          # 端数はそのまま（水増ししない）
        x, batch = data.batch(idx)
        pred = model(x, batch["f0"], temp=args.temp_lo, hard=True, noise=False)

        z = batch["z"].numpy()                                        # [B,M]
        g_pred = tf.argmax(pred["gate_logits"], -1,
                           output_type=tf.int32).numpy()              # [B,M]
        R_true = batch["R"].numpy().sum(axis=1)                       # [B,M]
        R_pred = pred["R"].numpy().sum(axis=1)                        # [B,M]

        e_true = ds.sources(batch["src_params"], batch["src_type"],
                            T_OFF, N, SR).numpy()[:, :, fidx]          # [B,K,T]
        e_param = ds.sources(pred["src_params"], pred["type_oh"][..., 0],
                             T_OFF, N, SR).numpy()[:, :, fidx]         # [B,K,T]
        traj = (pred["traj"].numpy() if "traj" in pred
                else np.zeros_like(e_true))

        t_true = batch["src_type"].numpy()                            # [B,K] 1=ENV
        t_pred = tf.argmax(pred["type_logits"], -1).numpy()           # 0=ENV
        t_pred = (1 - t_pred).astype(np.float32)                      # 1=ENV に揃える
        used = (z[:, :, None] == np.arange(K)[None, None, :]).any(axis=1)

        for k, v in (("idx", np.asarray(idx)), ("z_true", z), ("z_pred", g_pred),
                     ("d_true", R_true), ("d_pred", R_pred),
                     ("e_true", e_true), ("e_param", e_param), ("traj", traj),
                     ("type_true", t_true), ("type_pred", t_pred),
                     ("stat_true", batch["theta_stat"].numpy()),
                     ("stat_pred", pred["theta_stat"].numpy()),
                     ("used", used), ("f0", batch["f0"].numpy())):
            acc[k].append(v)
        if s0 % (args.batch * 20) == 0:
            print(f"  {s0 + len(idx)} / {len(idx_all)}", flush=True)

    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out["model"] = np.array(args.model)
    np.savez_compressed(args.out, **out)
    size = os.path.getsize(args.out) / 1e6
    print(f"saved {args.out}  ({len(out['idx'])} samples, {size:.1f} MB)")
    for k, v in out.items():
        if isinstance(v, np.ndarray) and v.ndim:
            print(f"  {k:10s} {str(v.shape):16s} {v.dtype}")


if __name__ == "__main__":
    main()
