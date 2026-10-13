#!/bin/bash
# 実験 37 の追加: 実際の軌跡で学習した主な 4 通りを、モデルの初期値の seed 1/2 でも
cd ~/modulation-assignment-synth
P=~/.venv/bin/python
for s in 1 2; do for c in "pool circ2" "pool cs" "ends circ2" "flat circ2"; do
  set -- $c; echo "$1 $2 $s"
done; done | xargs -P 8 -L 1 bash -c '$0 phase_probe.py --arch $1 --loss $2 --train traj --seed $3 --threads 1 --out probe_seed/$1_$2_traj_s$3.json > probe_seed/$1_$2_traj_s$3.log 2>&1' $P
echo done > probe_seed/done
