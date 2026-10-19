#!/bin/bash
# 実験 47: LFO の三角波を正しい式に直してデータセットを作り直す（dataset_v8）。
# 乱数の種・パラメータは dataset_v7 と同じ（変わるのは LFO を使う音の波形だけ）。途中から再開できる
cd ~/modulation-assignment-synth
export MOD_LAYOUT=mixed TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2
nice -n 10 ~/.venv/bin/python generate_dataset.py --n_total 100000 --out dataset_v8 --batch 64 --lfo_tri tri > gen_v8.log 2>&1
echo done > gen_v8.done
