r"""modspec_check.py — 変調スペクトル（実験 42）に LFO のレートが山として現れるかの確認

  MOD_LAYOUT=typed python modspec_check.py

test の音のうち、使用中の LFO が 1 本で、レート 1〜10 Hz、音量（level）・カットオフ・ピッチ（fin）の
どれかを |深さ| 0.5 以上で変調しているものについて、帯域を合計した変調スペクトルの 0.9 Hz 以上の最大の山の位置を、
真のレートと比べる（学習なしの単純な読み方）
"""
import numpy as np
import tensorflow as tf
import diff_synth_tf as ds
import estimator as E
from train import Data

D = Data("dataset_v7")
mel_mat = tf.signal.linear_to_mel_weight_matrix(64, 513, 16000, 30.0, 7800.0)
idx = D.splits["test"]
z, sp, R = D.P["z"][idx].astype(int), D.P["src_params"][idx], D.P["R"][idx]
used = np.stack([(z == k).any(1) for k in range(ds.K_ENVS)], 1)
n_lfo = used[:, 4:].sum(1)
rows = []
for dest_name, cols in (("level", [0, 1]), ("cutoff", [4]), ("fin", [2, 3])):
    sel = []
    for i in np.nonzero(n_lfo == 1)[0]:
        k = 4                                                   # LFO が 1 本なら枠 4
        r = sp[i, k, 0]
        dests = np.nonzero(z[i] == k)[0]
        if 1.0 <= r <= 10.0 and any(d in cols and abs(R[i, k, d]) >= 0.5 for d in dests):
            sel.append(i)
    sel = np.array(sel[:300])
    x, _ = D.batch(idx[sel])
    S = tf.abs(tf.signal.stft(x, 1024, 256, fft_length=1024))
    mel = tf.math.log(tf.matmul(S, mel_mat) + 1e-5)
    ms = E.mod_spectrum(mel).numpy()                            # [n,82,16]（log(1+振幅)）
    freqs = np.arange(1, E.MOD_BINS + 1) * 62.5 / E.MOD_NFFT
    prof = ms.sum(2)                                             # 帯域を合計
    lo = freqs >= 0.9                                            # ENV の成分（< 1 Hz）を除いて山を探す
    pk = freqs[lo][np.argmax(prof[:, lo], 1)]
    rt = sp[sel, 4, 0]
    rel = np.abs(pk - rt) / rt
    print(f"{dest_name:7s} {len(sel):4d} 音: 山の位置と真のレートの相関 {np.corrcoef(np.log(pk), np.log(rt))[0,1]:.3f}、"
          f"相対誤差の中央値 {np.median(rel):.3f}、±10% 以内 {np.mean(rel <= 0.1):.2f}、"
          f"倍・半分の取り違え {np.mean((np.abs(pk / rt - 2) < 0.2) | (np.abs(pk / rt - 0.5) < 0.1)):.2f}")
