"""
recover_legacy_ckpt.py — weights_io 導入前の ckpt から全重みを取り出し .weights.h5 にする
================================================================================
旧 ckpt は tf.train.Checkpoint(model, opt, step) で保存されており、
conv（リスト内で TF から辿れない）と BiGRU（opt 経由の方が経路が短い）の重みは
model/ ではなく opt/_trainable_variables/<i> の下に入っている。
opt を「学習時と同じく model.trainable_variables で build した状態」にして復元すれば全て戻る。
評価時に opt を build していなかったため、これまではこの部分が黙って復元されなかった。

  python recover_legacy_ckpt.py --model single --ckpt ckpt_nw0.25/ckpt-60
  python recover_legacy_ckpt.py --model two    --ckpt ckpt_two2/best/ckpt-68

出力 <ckpt>.weights.h5 は weights_io.latest() がそのまま拾うので、
以後は train.py / train2.py --eval_only や eval_dump.py で評価できる。
"""
import argparse, hashlib, json, os

import numpy as np
import tensorflow as tf

import estimator as E
import estimator2 as E2
import weights_io as W


def _h(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["single", "two"], required=True)
    ap.add_argument("--ckpt", required=True, help="ckpt の prefix（例: ckpt_nw0.25/ckpt-60）")
    ap.add_argument("--arch", default="v2")
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--data", default="dataset_v7")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.data, "manifest.json")))
    SR, N, T_OFF = man["sr"], man["N"], man["t_off"]
    model = E.Estimator(sr=SR, t_off=T_OFF, arch=args.arch) if args.model == "single" \
        else E2.TwoStageEstimator(sr=SR, t_off=T_OFF, hop=args.hop)
    model(tf.zeros([1, N]), tf.constant([220.0]), temp=1.0, hard=True)
    opt = tf.keras.optimizers.Adam(1e-3)
    opt.build(model.trainable_variables)         # 学習時の apply_gradients と同じ順序で build
    step = tf.Variable(0, dtype=tf.int64)
    tf.train.Checkpoint(model=model, opt=opt, step=step).restore(args.ckpt) \
        .assert_existing_objects_matched()

    reader = tf.train.load_checkpoint(args.ckpt)
    stored = {_h(reader.get_tensor(k)) for k in reader.get_variable_to_shape_map()
              if (k.startswith("model/") or k.startswith("opt/_trainable_variables/"))
              and "OPTIMIZER_SLOT" not in k}
    missing = [getattr(v, "path", v.name) for v in W._model_vars(model)
               if _h(v.numpy()) not in stored]
    if missing:
        raise SystemExit(f"復元できなかった変数 {len(missing)} 個: {missing}")

    out = W.ckpt_weights_path(args.ckpt)
    W.save(model, out)
    print(f"recovered {len(W._model_vars(model))} variables (step {int(step)}) → {out}")


if __name__ == "__main__":
    main()
