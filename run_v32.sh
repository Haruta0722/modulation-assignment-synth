#!/bin/bash
# 実験 46: v29（typed ＋ 変調スペクトル ＋ レートの分類）で、レートの分類の勾配だけ共有の層に流さず（--rate_ce_stop）、
#          レートの回帰を補助として共有の層まで流す（--rate_aux）。位相・morph も共有の層まで流す。seed 0/1/2
#   v32: 上記
#   v33: v32 ＋ 変調スペクトルに位相も加える（--mod_phase）
cd ~/modulation-assignment-synth
P=~/.venv/bin/python
export MOD_LAYOUT=typed TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1
one() { tag=$1; ex=$2; s=$3; C=ckpt_${tag}_single_s$s; A="--arch v4 --lfo_read mod --rate_head hlgauss $ex"
  $P train.py --data dataset_v7 $A --rate_ce_stop --rate_aux --steps 30000 --batch 32 --spec_warm0 999999 --none_w 0.25 \
     --depth_tf --src_w 30 --seed $s --ckpt_dir $C > run_${tag}_s$s.log 2>&1
  $P train.py --data dataset_v7 $A --rate_ce_stop --rate_aux --eval_only --split test --ckpt_dir $C > eval_${tag}_s$s.txt 2>&1
  $P eval_dump.py --model single $A --ckpt_dir $C --split test --out dump_${tag}_single_s${s}_test.npz > /dev/null 2>&1; }
for s in 0 1 2; do one v32 "" $s & one v33 "--mod_phase" $s & done
wait
for t in v32_single v33_single; do
  $P lfo_phase_check.py dump_${t}_s{0,1,2}_test.npz > lfo_${t}.txt 2>&1
  $P param_metrics.py dump_${t}_s{0,1,2}_test.npz > param_${t}.txt 2>&1
  $P desc_breakdown.py dump_${t}_s{0,1,2}_test.npz > descb_${t}.txt 2>&1
done
$P rate_precision.py v25 v29 v30 v32 v33 > rate_precision_v32.txt 2>&1
$P rate_probe.py --summary probe43 > probe43/summary_v32.txt 2>&1
echo done > run_v32.done
