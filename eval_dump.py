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
  gate_margin [B,M]    ゲートの「1位のソース」と「∅」のロジット差（正なら割り当て側）
  z_src       [B,M]    ∅ を除いた 1 位のソース（しきい値を下げたときの割り当て先）
  depth_raw   [B,M]    ゲートを掛ける前の深さ（tanh 出力）
  d_true      [B,M]    列ごとの真の深さ（R の列和）
  d_pred      [B,M]    同じく予測
  e_true      [B,K,T]  真のソース信号（評価フレーム上）
  e_param     [B,K,T]  予測パラメータから再合成したソース信号（shape_ok が使う量）
  traj        [B,K,T]  Stage 1 の生の軌跡（single では全て 0）
  type_true   [B,K]    1 = ENV, 0 = LFO
  type_pred   [B,K]
  sp_true     [B,K,5]  真のソースパラメータ（ENV: a,h,d,s,r / LFO: rate,phase,morph,0,0）
  sp_pred     [B,K,5]  予測のソースパラメータ（予測の型に応じて ENV / LFO の解釈）
  stat_true   [B,M]    静的値 θ_stat
  stat_pred   [B,M]
  used        [B,K]    そのソースがどこかの列で使われているか
  f0          [B]
"""
import argparse, os

# --- 決定性の設定: tensorflow を import する前に環境変数を立てる必要がある ---
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import numpy as np
import tensorflow as tf

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(0)

import diff_synth_tf as ds
import estimator as E
import estimator2 as E2
import train as T1
import weights_io as W

K, M = ds.K_ENVS, ds.M_DEST


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v7")
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--model", type=str, required=True, choices=["single", "two"])
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--arch", type=str, default="v2", choices=["v1", "v2", "v3", "v4"],
                    help="single のときのみ使用")
    ap.add_argument("--act", type=str, default="relu", choices=["relu", "leaky"],
                    help="single のときのみ使用。学習時と同じものを指定する")
    ap.add_argument("--lfo_read", type=str, default=None, choices=["trunk", "pool", "ends"],
                    help="学習時と同じものを指定する（既定は単段 trunk、二段 pool）")
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
        lr_ = args.lfo_read or "pool"
        W.check_config(args.ckpt_dir, dict(lfo_read=lr_))
        model = E2.TwoStageEstimator(sr=SR, t_off=T_OFF, hop=args.hop, lfo_read=lr_)
    else:
        lr_ = args.lfo_read or "trunk"
        W.check_config(args.ckpt_dir, dict(arch=args.arch, act=args.act, lfo_read=lr_))
        model = E.Estimator(sr=SR, t_off=T_OFF, arch=args.arch, act=args.act, lfo_read=lr_)
    model(tf.zeros([1, N]), tf.constant([220.0], tf.float32), temp=1.0, hard=True)

    W.load(model, W.latest(args.ckpt_dir))

    idx_all = data.splits[args.split]
    if args.n:
        idx_all = idx_all[:args.n]
    acc = {k: [] for k in ("idx", "z_true", "z_pred", "gate_margin", "z_src", "depth_raw",
                           "d_true", "d_pred",
                           "e_true", "e_param", "traj", "type_true", "type_pred",
                           "sp_true", "sp_pred",
                           "stat_true", "stat_pred", "used", "f0")}

    for s0 in range(0, len(idx_all), args.batch):
        idx = idx_all[s0:s0 + args.batch]          # 端数はそのまま（水増ししない）
        x, batch = data.batch(idx)
        pred = model(x, batch["f0"], temp=args.temp_lo, hard=True, noise=False)

        z = batch["z"].numpy()                                        # [B,M]
        g_pred = tf.argmax(pred["gate_logits"], -1,
                           output_type=tf.int32).numpy()              # [B,M]
        gl = pred["gate_logits"].numpy()                              # [B,M,K+1]
        margin = np.max(gl[..., :K], axis=-1) - gl[..., K]             # [B,M]
        z_src = np.argmax(gl[..., :K], axis=-1)                       # ∅ を除いた 1 位
        depth_raw = pred["depth"].numpy()                             # ゲート前の深さ
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
                     ("gate_margin", margin), ("z_src", z_src), ("depth_raw", depth_raw),
                     ("d_true", R_true), ("d_pred", R_pred),
                     ("e_true", e_true), ("e_param", e_param), ("traj", traj),
                     ("type_true", t_true), ("type_pred", t_pred),
                     ("sp_true", batch["src_params"].numpy()),
                     ("sp_pred", pred["src_params"].numpy()),
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
