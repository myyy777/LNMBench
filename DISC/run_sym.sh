#!/bin/bash
set -e

# 数据集、随机种子
DATASET="pathmnist"
SEED=0

# 所有任务列表
TASKS=(
  "--noise_type symmetric --noise_rate 0.2"
  "--noise_type symmetric --noise_rate 0.5"
  "--noise_type symmetric --noise_rate 0.8"
  "--noise_type symmetric --noise_rate 0.9"
)

mkdir -p logs
export CUDA_VISIBLE_DEVICES=0

for ((i=0; i<${#TASKS[@]}; i++)); do
  task="${TASKS[$i]}"
  NOISE_TYPE=$(echo $task | awk '{print $2}')
  NOISE_RATE=$(echo $task | awk '{print $4}')
  RATE_TAG=$(echo $NOISE_RATE | awk -F. '{printf "%s%s", $1, substr($2 "0",1,2)}')
  LOGFILE="logs/train_disc__${NOISE_TYPE}_seed${SEED}_rate${RATE_TAG}_50.log"

  echo ">>> [GPU 3] Running task $i: $task"
  echo ">>> [GPU 3] Logging to $LOGFILE"

  python -u main.py \
    --gpu 0 \
    --dataset $DATASET \
    --seed $SEED \
    $task \
    > "$LOGFILE" 2>&1

  echo ">>> [GPU 3] Finished task $i"
done

echo ">>> All tasks finished on GPU 3."
