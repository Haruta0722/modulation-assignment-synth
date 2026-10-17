#!/bin/bash
# 実験 43: 変調スペクトルだけから LFO のレートを学習する小さなモデル（LFO 1 本の音）。2 乗誤差 vs HL-Gauss、seed 0/1/2
cd ~/modulation-assignment-synth
P=~/.venv/bin/python
export MOD_LAYOUT=typed TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1
for l in mse hlgauss; do for s in 0 1 2; do
  $P rate_probe.py --loss $l --seed $s --threads 3 --out probe43/${l}_s$s.json > probe43/${l}_s$s.log 2>&1 &
done; done
wait
$P rate_probe.py --summary probe43 > probe43/summary.txt 2>&1
echo done > probe43/done
