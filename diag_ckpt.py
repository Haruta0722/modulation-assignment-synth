"""
diag_ckpt.py — 二段構成のチェックポイント復元を検査する
================================================================================
症状:
  学習中の val: F1 = 0.304 / traj_shape_ok = 0.076 / traj(train batch) = 0.23
  eval_only の test: F1 = 0.198 / traj_shape_ok = 0.0019 / traj_loss = 0.43

  traj_loss 0.43 は学習開始時（0.469）とほぼ同じ。
  → Stage 1 の重みが復元されていない可能性。

本スクリプトの確認項目:
  1. restore の consumed 状態（assert_consumed で未復元変数を検出）
  2. 復元前後で Stage 1 の出力が変わるか（変わらなければ復元が効いていない）
  3. 同一の復元済みモデルで val と test を評価（差があればデータ側の問題）

使い方:
  python diag_ckpt.py --data dataset_v7 --ckpt_dir ckpt_two
"""
import argparse
import numpy as np
import tensorflow as tf

import diff_synth_tf as ds
import estimator as E
import estimator2 as E2
import train as T1

K, M = ds.K_ENVS, ds.M_DEST
LAM = dict(spec=0.0, param=0.2, assign=3.0, sparse=0.0, none_w=0.25, traj=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v7")
    ap.add_argument("--ckpt_dir", type=str, default="ckpt_two")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--hop", type=int, default=256)
    args = ap.parse_args()

    data = T1.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    fmax = E.F0_LO * 2 ** (100 / 1200)
    NH = (int((SR / 2) // fmax), int(((SR / 2) / fmax - 1) // 2))
    n_frames = 1 + (N - 1024) // args.hop
    fidx = np.minimum(np.arange(n_frames) * args.hop + 512, N - 1)

    model = E2.TwoStageEstimator(sr=SR, t_off=T_OFF, hop=args.hop)
    model(tf.zeros([1, N]), tf.constant([220.0], tf.float32), temp=1.0, hard=True)
    n_vars = len(model.trainable_variables)
    print(f"モデル構築完了: 学習可能変数 {n_vars} 個")
    print(f"  stage1: {len(model.stage1.trainable_variables)} 個")
    print(f"  stage2: {len(model.stage2.trainable_variables)} 個")

    # --- 復元前の Stage 1 出力を記録 ---
    x0, b0 = data.batch(data.splits["test"][:8])
    traj_before = model.stage1(x0, b0["f0"]).numpy()

    # --- 復元 ---
    latest = tf.train.latest_checkpoint(args.ckpt_dir)
    print(f"\nlatest checkpoint = {latest}")
    if latest is None:
        print("  → チェックポイントが見つかりません。--ckpt_dir を確認してください。")
        return
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(latest)
    try:
        status.assert_existing_objects_matched()
        print("  assert_existing_objects_matched: OK")
    except Exception as e:
        print(f"  assert_existing_objects_matched: 失敗\n    {str(e)[:400]}")
    try:
        status.assert_consumed()
        print("  assert_consumed: OK（全変数が復元された）")
    except Exception as e:
        print(f"  assert_consumed: 失敗（未復元の変数がある）\n    {str(e)[:400]}")

    # --- 復元後の Stage 1 出力と比較 ---
    traj_after = model.stage1(x0, b0["f0"]).numpy()
    delta = float(np.abs(traj_after - traj_before).max())
    print(f"\nStage1 出力の変化: max|Δ| = {delta:.6f}")
    if delta < 1e-8:
        print("  → **復元前後で出力が同一。Stage 1 の重みが復元されていない**")
    else:
        print("  → 出力が変化。Stage 1 は復元されている")

    # --- チェックポイント内のキーを確認 ---
    print("\nチェックポイント内の変数（stage1 関連の先頭 8 件）:")
    reader = tf.train.load_checkpoint(latest)
    keys = sorted(reader.get_variable_to_shape_map().keys())
    s1keys = [k for k in keys if "stage1" in k]
    for k in s1keys[:8]:
        print(f"  {k}")
    print(f"  （stage1 関連は計 {len(s1keys)} 件 / 全 {len(keys)} 件）")
    if not s1keys:
        print("  → **チェックポイントに stage1 の変数が保存されていない**")

    # --- 同一モデルで val と test を評価 ---
    def evaluate(split):
        idx_all = data.splits[split][:args.n]
        agg, nb = None, 0
        for s0 in range(0, len(idx_all), args.batch):
            idx = idx_all[s0:s0 + args.batch]
            if len(idx) < args.batch:
                idx = np.concatenate([idx, idx_all[:args.batch - len(idx)]])
            x, batch = data.batch(idx)
            pred = model(x, batch["f0"], temp=0.3, hard=True, noise=False)
            total, parts, mapping = E2.losses2(pred, batch, x, SR, T_OFF, NH,
                                               LAM, args.hop)
            m = T1.metrics_np(pred, batch, mapping, SR, T_OFF, N)
            e_true = ds.sources(batch["src_params"], batch["src_type"],
                                T_OFF, N, SR).numpy()[:, :, fidx]
            z = batch["z"].numpy()
            used = np.stack([(z == k).any(axis=1) for k in range(K)], 1)
            err = np.abs(pred["traj"].numpy() - e_true).mean(-1)
            m["traj_shape_ok"] = float((err[used] <= 0.1).mean())
            m["traj_loss"] = float(parts["traj"])
            agg = {k: agg.get(k, 0) + v for k, v in m.items()} if agg else dict(m)
            nb += 1
        return {k: v / nb for k, v in agg.items()}

    print(f"\n同一モデルでの評価（各 {args.n} サンプル）:")
    print(f"{'指標':20s} {'val':>10s} {'test':>10s}")
    rv, rt = evaluate("val"), evaluate("test")
    for k in ("F1", "F1_core8", "desc_rec_rate", "traj_shape_ok",
              "traj_loss", "shape_ok_on_hit", "n_pred"):
        print(f"{k:20s} {rv.get(k, float('nan')):10.4f} {rt.get(k, float('nan')):10.4f}")
    print("\n  val と test が近ければデータ側の問題はない。")
    print("  両方とも学習中の val（F1 0.304 / traj_ok 0.076）より低ければ復元の問題。")


if __name__ == "__main__":
    main()
