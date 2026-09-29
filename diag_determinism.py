"""
diag_determinism.py — 単段 eval_only が実行ごとに揺れる原因の切り分け
================================================================================
同じコマンドを 2 回実行し、末尾の FINGERPRINT 行を比べる。

  python diag_determinism.py --data dataset_v7 --ckpt_dir ckpt_nw0.25

判定:
  missing>0 / restored が実行間で違う  → 復元漏れ（初期化値が残っている変数がある）
  restored は同じ & data が違う        → Data 側（splits / 正準順序）の問題
  restored・data が同じ & logits が違う → forward の非決定性
  全部同じ & eval_only の値が揺れる     → 評価ループ側（集計・バッチ構成）の問題
"""
import argparse, hashlib, os

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import numpy as np
import tensorflow as tf

import estimator as E
import train as T1


def h(*arrays):
    m = hashlib.md5()
    for a in arrays:
        m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()[:10]


def var_name(v):
    return getattr(v, "path", None) or v.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_v7")
    ap.add_argument("--ckpt_dir", type=str, default="ckpt_nw0.25")
    ap.add_argument("--arch", type=str, default="v2")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    try:
        import keras; kv = keras.__version__
    except Exception:
        kv = "?"
    print(f"tensorflow {tf.__version__} / keras {kv}")

    data = T1.Data(args.data)
    SR, N, T_OFF = data.man["sr"], data.man["N"], data.man["t_off"]
    sp = data.splits[args.split]
    idx_all = sp[:args.n]
    data_fp = h(idx_all, data.P["z"][idx_all], data.P["src_params"][idx_all],
                data.wav[idx_all])
    print(f"split {args.split}: [{sp[0]}, {sp[-1]}] n={len(sp)}")

    model = E.Estimator(sr=SR, t_off=T_OFF, arch=args.arch)
    model(tf.zeros([1, N]), tf.constant([220.0], tf.float32), temp=1.0, hard=True)
    ordered = sorted(model.variables, key=var_name)
    init_fp = h(*[v.numpy() for v in ordered])
    print(f"model variables: {len(ordered)}")

    path = tf.train.latest_checkpoint(args.ckpt_dir)
    if path is None:
        raise SystemExit(f"チェックポイントが見つかりません: {os.path.abspath(args.ckpt_dir)}")
    print(f"checkpoint: {path}")
    status = tf.train.Checkpoint(model=model).restore(path)
    try:
        status.assert_existing_objects_matched()
        print("  assert_existing_objects_matched: OK")
    except AssertionError as e:
        print(f"  assert_existing_objects_matched: 失敗\n    {str(e)[:600]}")
    restored_fp = h(*[v.numpy() for v in ordered])

    # 名前に依存せず、各変数の値が ckpt 内のどれかの配列と完全一致するかで復元を確かめる
    reader = tf.train.load_checkpoint(path)
    stored = set()
    n_stored = 0
    for k in reader.get_variable_to_shape_map():
        if not k.startswith("model/") or "OPTIMIZER_SLOT" in k:
            continue
        a = reader.get_tensor(k)
        if isinstance(a, np.ndarray) and a.dtype.kind == "f":
            stored.add(h(a.astype(np.float32)))
            n_stored += 1
    missing = [v for v in ordered if h(v.numpy().astype(np.float32)) not in stored]
    print(f"  ckpt 内のモデル配列 {n_stored} 個 / モデル変数 {len(ordered)} 個")
    if missing:
        print(f"  **ckpt の値と一致しない変数 {len(missing)} 個（初期化値のまま）**:")
        for v in missing[:20]:
            print(f"    {var_name(v)} {tuple(v.shape)}")
    else:
        print("  全変数が ckpt の値と一致（復元は完全）")

    def logits_pass():
        outs = []
        for s0 in range(0, len(idx_all), args.batch):
            x, b = data.batch(idx_all[s0:s0 + args.batch])
            p = model(x, b["f0"], temp=0.3, hard=True, noise=False)
            outs.append(p["gate_logits"].numpy())
        return np.concatenate(outs)

    g1, g2 = logits_pass(), logits_pass()
    print(f"同一プロセス 2 回の gate_logits: max|Δ| = {np.abs(g1 - g2).max():.3e}")

    K = g1.shape[-1] - 1
    top = g1.argmax(-1)
    margin = g1[..., :K].max(-1) - g1[..., K]        # >0 ならソースあり
    print(f"n_pred(平均) = {(top < K).sum(-1).mean():.4f}   "
          f"|gate_margin|<0.1 の列: {(np.abs(margin) < 0.1).mean():.2%}")

    print(f"\nFINGERPRINT init={init_fp} restored={restored_fp} data={data_fp} "
          f"logits={h(g1)} missing={len(missing)}")


if __name__ == "__main__":
    main()
