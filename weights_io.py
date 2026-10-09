"""
weights_io.py — モデル重みの保存・読み込み（検証つき）
================================================================================
Keras 3 では tf.train.Checkpoint がリストに入れた層（self.convs）や
Bidirectional(GRU) を保存対象として辿らず、エンコーダの重みが黙って欠落していた。
重みの正本は Keras の save_weights（.weights.h5）とし、保存直後と読み込み直後に
「全変数の値がファイル内のいずれかの配列と完全一致するか」を名前に依存せず照合する。
tf.train.Checkpoint はオプティマイザの状態とステップ数（学習再開用）にだけ使う。
"""
import glob, hashlib, os

import h5py
import numpy as np

SUFFIX = ".weights.h5"


def _h(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).hexdigest()


def _model_vars(model):
    # seed_generator の状態は乱数の内部状態で、学習結果ではない
    return [v for v in model.variables if "seed_generator" not in (getattr(v, "path", "") or v.name)]


def _file_hashes(path):
    out = set()
    with h5py.File(path, "r") as f:
        f.visititems(lambda _, o: out.add(_h(o[()])) if isinstance(o, h5py.Dataset)
                     and o.dtype.kind == "f" else None)
    return out


def verify(model, path):
    stored = _file_hashes(path)
    bad = [getattr(v, "path", v.name) for v in _model_vars(model)
           if _h(v.numpy()) not in stored]
    if bad:
        raise RuntimeError(f"{path}: モデルの値と一致しない変数が {len(bad)} 個あります: {bad[:10]}")


def save(model, path):
    assert path.endswith(SUFFIX), path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path[:-len(SUFFIX)] + ".tmp" + SUFFIX
    model.save_weights(tmp)
    verify(model, tmp)
    os.replace(tmp, path)


def load(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"重みファイルがありません: {path}（Keras 3 移行前の ckpt は重みが欠落しているため使えません）")
    model.load_weights(path)
    verify(model, path)
    print(f"weights loaded & verified: {path} ({len(_model_vars(model))} variables)")


def check_config(ckpt_dir, cfg, write=False):
    """構造・活性化関数など、重みの照合では分からない設定を ckpt と突き合わせる。
    活性化関数が違っても重みは読み込めてしまい、出力だけが変わるため"""
    import json
    path = os.path.join(ckpt_dir, "model_config.json")
    if write:
        os.makedirs(ckpt_dir, exist_ok=True)
        if os.path.exists(path):
            check_config(ckpt_dir, cfg)
        else:
            json.dump(cfg, open(path, "w"), indent=1)
        return
    if not os.path.exists(path):
        return                                    # 設定の記録を始める前の ckpt
    saved = json.load(open(path))
    diff = {k: (saved.get(k), v) for k, v in cfg.items() if k in saved and saved[k] != v}
    if diff:
        raise SystemExit(f"{path} の設定と一致しません（保存時, 今回）: {diff}")


def ckpt_weights_path(ckpt_prefix):
    """CheckpointManager の保存先（…/ckpt-60）に対応する重みファイル"""
    return ckpt_prefix + SUFFIX


def prune(mgr):
    """CheckpointManager が消した古い ckpt に対応する重みファイルを消す"""
    keep = {ckpt_weights_path(p) for p in mgr.checkpoints}
    for p in glob.glob(os.path.join(mgr.directory, "ckpt-*" + SUFFIX)):
        if p not in keep:
            os.remove(p)


def latest(ckpt_dir):
    """評価用: best/ があればそれを、なければ最新の重みファイルを返す"""
    import tensorflow as tf
    for d in (os.path.join(ckpt_dir, "best"), ckpt_dir):
        p = tf.train.latest_checkpoint(d)
        if p and os.path.exists(ckpt_weights_path(p)):
            return ckpt_weights_path(p)
    raise FileNotFoundError(f"{ckpt_dir} に検証済みの重みファイル（*{SUFFIX}）がありません")
