#!/bin/bash
set -e

# 数据集
DATASET="organcmnist"

# 种子列表
SEEDS=(0 5 10)

# 所有任务列表
TASKS=(
  "--noise_type instance --noise_rate 0.2"
  "--noise_type instance --noise_rate 0.5"
  "--noise_type instance --noise_rate 0.9"
)

mkdir -p logs
export CUDA_VISIBLE_DEVICES=3

for SEED in "${SEEDS[@]}"; do
  for ((i=0; i<${#TASKS[@]}; i++)); do
    task="${TASKS[$i]}"
    NOISE_TYPE=$(echo $task | awk '{print $2}')
    NOISE_RATE=$(echo $task | awk '{print $4}')
    RATE_TAG=$(echo $NOISE_RATE | awk -F. '{printf "%s%s", $1, substr($2 "0",1,2)}')
    LOGFILE="logs/train_disc_${DATASET}__${NOISE_TYPE}_seed${SEED}_rate${RATE_TAG}_50.log"

    echo ">>> [GPU 3] Running seed $SEED, task $i: $task"
    echo ">>> [GPU 3] Logging to $LOGFILE"

    python -u main.py \
      --gpu 0 \
      --dataset $DATASET \
      --seed $SEED \
      $task \
      > "$LOGFILE" 2>&1

    echo ">>> [GPU 3] Finished seed $SEED, task $i"
  done
done

echo ">>> All tasks finished on GPU 3."
