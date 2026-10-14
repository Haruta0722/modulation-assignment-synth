#!/bin/bash
# 実験 40（診断）: Stage 1 に正解の枠の情報を渡して、軌跡の崩れ（揺れの縮み・ENV の漏れ込み）が消えるか
#   v24t: v22s ＋ --oracle type （各枠が ENV か LFO か、使われているか）
#   v24r: v22s ＋ --oracle route（それに加えて各枠の変調先）
# seed 0 は 6 スレッドで先に、seed 1/2 は 2 スレッド・優先度を下げて並行
cd ~/modulation-assignment-synth
P=~/.venv/bin/python
run() { tag=$1; o=$2; s=$3; th=$4; C=ckpt_${tag}_two_s$s
  export TF_NUM_INTRAOP_THREADS=$th TF_NUM_INTEROP_THREADS=1
  A="--lfo_read ends --s2_stopgrad --oracle $o"
  $P train2.py --data dataset_v7 --steps 30000 --batch 32 --depth_tf --seed $s $A --ckpt_dir $C > run_${tag}_two_s$s.log 2>&1
  $P train2.py --data dataset_v7 --eval_only --split test $A --ckpt_dir $C > eval_${tag}_two_s$s.txt 2>&1
  $P eval_dump.py --model two --lfo_read ends --oracle $o --ckpt_dir $C --split test --out dump_${tag}_two_s${s}_test.npz > /dev/null 2>&1
  echo done > ${C}.done; }
run v24t type 0 6 &
run v24r route 0 6 &
nice -n 10 bash -c "$(declare -f run); P=$P; run v24t type 1 2 & run v24r route 1 2 & run v24t type 2 2 & run v24r route 2 2 & wait" &
wait
for t in v24t_two v24r_two; do
  $P lfo_phase_check.py dump_${t}_s{0,1,2}_test.npz > lfo_${t}.txt 2>&1
  $P traj_slot_check.py dump_${t}_s{0,1,2}_test.npz > slot_${t}.txt 2>&1
  $P param_metrics.py dump_${t}_s{0,1,2}_test.npz > param_${t}.txt 2>&1
done
$P traj_slot_check.py dump_v22s_two_s{0,1,2}_test.npz > slot_v22s_two.txt 2>&1
echo done > run_v24.done
