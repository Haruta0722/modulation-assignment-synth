#!/bin/bash
# 実験 48: LFO の三角波を直したデータセット（dataset_v8）で学習し直す。typed、単段、seed 0/1/2
#   v34: v25 と同じ設定（基準）
#   v35: v32 と同じ設定（変調スペクトル ＋ レートの分類、分類の勾配だけ共有の層に流さない、補助のレートの回帰）
cd ~/modulation-assignment-synth
P=~/.venv/bin/python
export MOD_LAYOUT=typed TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1
one() { tag=$1; tr=$2; ev=$3; s=$4; C=ckpt_${tag}_single_s$s
  $P train.py --data dataset_v8 --arch v4 $tr --steps 30000 --batch 32 --spec_warm0 999999 --none_w 0.25 --depth_tf \
     --src_w 30 --seed $s --ckpt_dir $C > run_${tag}_s$s.log 2>&1
  $P train.py --data dataset_v8 --arch v4 $tr --eval_only --split test --ckpt_dir $C > eval_${tag}_s$s.txt 2>&1
  $P eval_dump.py --data dataset_v8 --model single --arch v4 $ev --ckpt_dir $C --split test --out dump_${tag}_single_s${s}_test.npz > /dev/null 2>&1; }
V35T="--lfo_read mod --rate_head hlgauss --rate_ce_stop --rate_aux"
V35E="--lfo_read mod --rate_head hlgauss"
for s in 0 1 2; do one v34 "" "" $s & one v35 "$V35T" "$V35E" $s & done
wait
for t in v34_single v35_single; do
  $P lfo_phase_check.py dump_${t}_s{0,1,2}_test.npz > lfo_${t}.txt 2>&1
  $P param_metrics.py dump_${t}_s{0,1,2}_test.npz > param_${t}.txt 2>&1
  $P desc_breakdown.py dump_${t}_s{0,1,2}_test.npz > descb_${t}.txt 2>&1
  $P lfo_whatif.py dump_${t}_s{0,1,2}_test.npz > whatif_${t}.txt 2>&1
done
$P rate_precision.py v25 v32 v34 v35 > rate_precision_v34.txt 2>&1
$P rate_probe.py --summary probe43 > probe43/summary_v34.txt 2>&1
echo done > run_v34.done
