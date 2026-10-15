#!/bin/bash
# 実験 42: LFO のヘッドに変調スペクトルの枝を加える（--lfo_read mod）。v25（typed、単段）との違いはこの点だけ。seed 0/1/2
cd ~/modulation-assignment-synth
P=~/.venv/bin/python
export MOD_LAYOUT=typed TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=1
one() { s=$1; C=ckpt_v27_single_s$s
  $P train.py --data dataset_v7 --arch v4 --steps 30000 --batch 32 --spec_warm0 999999 --none_w 0.25 --depth_tf \
     --src_w 30 --lfo_read mod --seed $s --ckpt_dir $C > run_v27_s$s.log 2>&1
  $P train.py --data dataset_v7 --arch v4 --lfo_read mod --eval_only --split test --ckpt_dir $C > eval_v27_s$s.txt 2>&1
  $P eval_dump.py --model single --arch v4 --lfo_read mod --ckpt_dir $C --split test --out dump_v27_single_s${s}_test.npz > /dev/null 2>&1; }
one 0 & one 1 & one 2 & wait
for t in v25_single v27_single; do
  $P desc_breakdown.py dump_${t}_s{0,1,2}_test.npz > descb_${t}.txt 2>&1
done
$P lfo_phase_check.py dump_v27_single_s{0,1,2}_test.npz > lfo_v27_single.txt 2>&1
$P param_metrics.py dump_v27_single_s{0,1,2}_test.npz > param_v27_single.txt 2>&1
echo done > run_v27.done
