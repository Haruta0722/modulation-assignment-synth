#!/bin/bash
# 実験 41: ソースの枠を ENV 4 枠 ＋ LFO 3 枠に型で分ける（MOD_LAYOUT=typed）。seed 0/1/2
#   v25: 単段 v18 と同じ設定（--arch v4 --depth_tf --src_w 30）＋ typed
#   v26: 二段 v22s と同じ設定（--depth_tf --lfo_read ends --s2_stopgrad）＋ typed
cd ~/modulation-assignment-synth
P=~/.venv/bin/python
export MOD_LAYOUT=typed TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
single() { s=$1; C=ckpt_v25_single_s$s
  $P train.py --data dataset_v7 --arch v4 --steps 30000 --batch 32 --spec_warm0 999999 --none_w 0.25 --depth_tf \
     --src_w 30 --seed $s --ckpt_dir $C > run_v25_s$s.log 2>&1
  $P train.py --data dataset_v7 --arch v4 --eval_only --split test --ckpt_dir $C > eval_v25_s$s.txt 2>&1
  $P eval_dump.py --model single --arch v4 --ckpt_dir $C --split test --out dump_v25_single_s${s}_test.npz > /dev/null 2>&1; }
two() { s=$1; C=ckpt_v26_two_s$s; A="--lfo_read ends --s2_stopgrad"
  $P train2.py --data dataset_v7 --steps 30000 --batch 32 --depth_tf --seed $s $A --ckpt_dir $C > run_v26_two_s$s.log 2>&1
  $P train2.py --data dataset_v7 --eval_only --split test $A --ckpt_dir $C > eval_v26_two_s$s.txt 2>&1
  $P eval_dump.py --model two --lfo_read ends --ckpt_dir $C --split test --out dump_v26_two_s${s}_test.npz > /dev/null 2>&1; }
for s in 0 1 2; do single $s & two $s & done
wait
for t in v25_single v26_two; do
  $P lfo_phase_check.py dump_${t}_s{0,1,2}_test.npz > lfo_${t}.txt 2>&1
  $P param_metrics.py dump_${t}_s{0,1,2}_test.npz > param_${t}.txt 2>&1
done
$P traj_slot_check.py dump_v26_two_s{0,1,2}_test.npz > slot_v26_two.txt 2>&1
echo done > run_v25.done
