#!/usr/bin/env bash
# Evaluate the released checkpoints. Loads frozen weights, recomputes the F1
# threshold on dev, evaluates test, and prints "TEST set prediction results".
#
#   export DATA_DIR=/path/to/data
#   export PYTHON=python            # env with torch + h5py
#   bash reproduce.sh               # all 7
#   bash reproduce.sh window_chb_K3 # one
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/configs.sh"
PY="${PYTHON:-python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT_ROOT="${OUT_ROOT:-$HERE/_reproduce_out}"

run_one () {
  local name="$1"
  local ckpt="$HERE/ckpts/$name/best.pth.tar"
  [ -f "$ckpt" ] || { echo "!! missing checkpoint: $ckpt"; return 1; }
  mkdir -p "$OUT_ROOT/$name"
  echo ">>> $name"
  # run from src/ so relative data paths resolve
  ( cd "$HERE/src" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" main.py \
      --test --load_model_path "$ckpt" --rand_seed 123 \
      --save_dir "$OUT_ROOT/$name" $("CFG_$name") )
}

[ "$#" -gt 0 ] && targets=("$@") || targets=("${ALL_CONFIGS[@]}")
for name in "${targets[@]}"; do run_one "$name"; done
