#!/bin/bash
# 実験 38: LFO のヘッドを BiGRU の両端の状態から読む（実験 37 の ends）。seed 0/1/2
#   v22 : 二段 v11 ＋ --lfo_read ends
#   v22s: 二段 v11 ＋ --lfo_read ends --s2_stopgrad（Stage 2 の勾配を Stage 1 に流さない）
#   v23 : 単段 v18 ＋ --lfo_read ends
cd ~/modulation-assignment-synth
P=~/.venv/bin/python
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
two() { tag=$1; s=$2; shift 2; C=ckpt_${tag}_two_s$s
  $P train2.py --data dataset_v7 --steps 30000 --batch 32 --depth_tf --seed $s "$@" --ckpt_dir $C > run_${tag}_two_s$s.log 2>&1
  $P train2.py --data dataset_v7 --eval_only --split test "$@" --ckpt_dir $C > eval_${tag}_two_s$s.txt 2>&1
  $P eval_dump.py --model two --lfo_read ends --ckpt_dir $C --split test --out dump_${tag}_two_s${s}_test.npz > /dev/null 2>&1; }
single() { s=$1; C=ckpt_v23_single_s$s
  $P train.py --data dataset_v7 --arch v4 --steps 30000 --batch 32 --spec_warm0 999999 --none_w 0.25 --depth_tf \
     --src_w 30 --lfo_read ends --seed $s --ckpt_dir $C > run_v23_s$s.log 2>&1
  $P train.py --data dataset_v7 --arch v4 --lfo_read ends --eval_only --split test --ckpt_dir $C > eval_v23_s$s.txt 2>&1
  $P eval_dump.py --model single --arch v4 --lfo_read ends --ckpt_dir $C --split test --out dump_v23_single_s${s}_test.npz > /dev/null 2>&1; }
for s in 0 1 2; do
  two v22 $s --lfo_read ends &
  two v22s $s --lfo_read ends --s2_stopgrad &
  single $s &
done
wait
for t in v22_two v22s_two v23_single; do
  $P lfo_phase_check.py dump_${t}_s{0,1,2}_test.npz > lfo_${t}.txt 2>&1
  $P param_metrics.py dump_${t}_s{0,1,2}_test.npz > param_${t}.txt 2>&1
done
echo done > run_v22.done
