#!/bin/bash
# 実験 49: dataset_v8 で二段（typed、v26 と同じ設定: --lfo_read ends --s2_stopgrad）を学習する。seed 0/1/2
#   目的: 二段の Stage 1 の軌跡と v35（単段）の精密なレートを組み合わせ、型紙の当てはめで LFO の位相を求める
cd ~/modulation-assignment-synth
P=~/.venv/bin/python
export MOD_LAYOUT=typed TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=1
one() { s=$1; C=ckpt_v36_two_s$s; A="--lfo_read ends --s2_stopgrad"
  $P train2.py --data dataset_v8 --steps 30000 --batch 32 --depth_tf --seed $s $A --ckpt_dir $C > run_v36_two_s$s.log 2>&1
  $P train2.py --data dataset_v8 --eval_only --split test $A --ckpt_dir $C > eval_v36_two_s$s.txt 2>&1
  $P eval_dump.py --data dataset_v8 --model two --lfo_read ends --ckpt_dir $C --split test --out dump_v36_two_s${s}_test.npz > /dev/null 2>&1; }
one 0 & one 1 & one 2 & wait
$P lfo_phase_check.py dump_v36_two_s{0,1,2}_test.npz > lfo_v36_two.txt 2>&1
$P traj_slot_check.py dump_v36_two_s{0,1,2}_test.npz > slot_v36_two.txt 2>&1
$P desc_breakdown.py dump_v36_two_s{0,1,2}_test.npz > descb_v36_two.txt 2>&1
echo done > run_v36.done
