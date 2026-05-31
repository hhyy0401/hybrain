#!/usr/bin/env bash
# Train the released configs from scratch (same flags, minus --test/--load).
# Writes best.pth.tar to each run dir and logs "TEST set prediction results".
#
#   export DATA_DIR=/path/to/data
#   export PYTHON=python
#   bash train.sh                                  # all configs, seed 123
#   bash train.sh window_chb_K3                    # one config
#   SEEDS="123 456 789" bash train.sh window_chb_K3  # paper mean +/- std
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/configs.sh"
PY="${PYTHON:-python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT_ROOT="${OUT_ROOT:-$HERE/runs}"
SEEDS="${SEEDS:-123}"

train_one () {
  local name="$1" seed="$2"
  mkdir -p "$OUT_ROOT/${name}_s${seed}"
  echo ">>> train $name (seed $seed)"
  ( cd "$HERE/src" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" main.py \
      --rand_seed "$seed" --save_dir "$OUT_ROOT/${name}_s${seed}" $("CFG_$name") )
}

[ "$#" -gt 0 ] && targets=("$@") || targets=("${ALL_CONFIGS[@]}")
for name in "${targets[@]}"; do
  for seed in $SEEDS; do train_one "$name" "$seed"; done
done
