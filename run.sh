#!/usr/bin/env bash
set -e  


NOISE_RATE="$1"   # 
NOISE_TYPE="$2"   # 
DATASET="$3"      #

if [ -z "$NOISE_RATE" ] || [ -z "$NOISE_TYPE" ] || [ -z "$DATASET" ]; then
  echo  $0 <noise_rate> <noise_type> <dataset>"
  echo "$0 0.2 symmetric pathmnist"
  exit 1
fi


for dir in */; do

  if [ ! -f "${dir}main.py" ]; then
    echo "skip ${dir} (nomain.py)"
    continue
  fi

  echo "=============================="
  echo "运行 ${dir}main.py"
  echo "noise_rate=$NOISE_RATE  noise_type=$NOISE_TYPE  dataset=$DATASET"
  echo "=============================="

  python "${dir}main.py" \
    --noise_rate "$NOISE_RATE" \
    --noise_type "$NOISE_TYPE" \
    --dataset "$DATASET"
done
