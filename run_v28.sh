#!/bin/bash
# 実験 44: LFO のレートを HL-Gauss の分類で出す（--rate_head hlgauss）。typed、単段、seed 0/1/2
#   v28: v25 ＋ 分類（変調スペクトルなし）
#   v29: v27 ＋ 分類（変調スペクトルあり、--lfo_read mod）
cd ~/modulation-assignment-synth
P=~/.venv/bin/python
export MOD_LAYOUT=typed TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1
one() { tag=$1; lr=$2; s=$3; C=ckpt_${tag}_single_s$s; A="--arch v4 --lfo_read $lr --rate_head hlgauss"
  $P train.py --data dataset_v7 $A --steps 30000 --batch 32 --spec_warm0 999999 --none_w 0.25 --depth_tf \
     --src_w 30 --seed $s --ckpt_dir $C > run_${tag}_s$s.log 2>&1
  $P train.py --data dataset_v7 $A --eval_only --split test --ckpt_dir $C > eval_${tag}_s$s.txt 2>&1
  $P eval_dump.py --model single $A --ckpt_dir $C --split test --out dump_${tag}_single_s${s}_test.npz > /dev/null 2>&1; }
for s in 0 1 2; do one v28 trunk $s & one v29 mod $s & done
wait
for t in v28_single v29_single; do
  $P lfo_phase_check.py dump_${t}_s{0,1,2}_test.npz > lfo_${t}.txt 2>&1
  $P param_metrics.py dump_${t}_s{0,1,2}_test.npz > param_${t}.txt 2>&1
  $P desc_breakdown.py dump_${t}_s{0,1,2}_test.npz > descb_${t}.txt 2>&1
done
$P rate_precision.py v25 v27 v28 v29 > rate_precision_v28.txt 2>&1
$P rate_probe.py --summary probe43 > probe43/summary_v28.txt 2>&1
echo done > run_v28.done
