#!/usr/bin/env bash
# Flag definitions for the 7 released checkpoints. Sourced by train.sh and
# reproduce.sh. Set DATA_DIR (parent of tusz_*/ and chb_mit*/) before sourcing.

: "${DATA_DIR:?Set DATA_DIR to the parent folder of your TUSZ / CHB-MIT data}"

# Raw EDF trees (override if not at the default location).
TUSZ_RAW="${TUSZ_RAW:-$DATA_DIR/tusz_v2.0.6/edf}"
CHB_RAW="${CHB_RAW:-$DATA_DIR/chb_mit}"
TUSZ_INPUT="--input_dir $DATA_DIR/tusz_resampled --raw_data_dir $TUSZ_RAW"
CHB_INPUT="--input_dir $DATA_DIR/chb_mit_resampled --raw_data_dir $CHB_RAW"

# Shared flags. num_workers=0: prediction loader is not multiprocessing-safe.
BASE="--graph_type none --use_fft --use_node_emb --n_hyper_layers 2 \
      --dropout 0.0 --l2_wd 5e-4 --metric_name auroc --data_augment --num_workers 0"

# Temporal attention + aux head (detection only).
TATTN="--temporal_attn --tattn_pos_enc --tattn_position after --aux_type bce --aux_weight 0.3"

ALL_CONFIGS=(
  window_tusz12 window_chb_K1 window_chb_K3
  pointwise_tusz60 pointwise_chb
  prediction_tusz12_K2 prediction_tusz60_K1
)

# --- window-based detection ---
CFG_window_tusz12 () {
  echo "--dataset TUSZ --task detection --model_name light_st_hyper2 \
        --num_nodes 19 --max_seq_len 12 --time_step_size 1 --no_bidirectional \
        --rnn_units 128 --n_hyperedges 1 --tattn_n_layers 2 \
        --num_epochs 80 --patience 10 --train_batch_size 128 --test_batch_size 256 \
        --lr_init 1e-3 --eval_every 3 $BASE $TATTN $TUSZ_INPUT"
}
CFG_window_chb_K1 () {
  echo "--dataset CHBMIT --task detection --model_name light_st_hyper2 \
        --num_nodes 22 --max_seq_len 12 --time_step_size 1 --no_bidirectional \
        --rnn_units 128 --n_hyperedges 1 --tattn_n_layers 1 \
        --num_epochs 100 --patience 15 --train_batch_size 128 --test_batch_size 256 \
        --lr_init 1e-3 --eval_every 3 $BASE $TATTN $CHB_INPUT"
}
CFG_window_chb_K3 () {
  echo "--dataset CHBMIT --task detection --model_name light_st_hyper2 \
        --num_nodes 22 --max_seq_len 12 --time_step_size 1 --no_bidirectional \
        --rnn_units 128 --n_hyperedges 3 --tattn_n_layers 1 \
        --num_epochs 100 --patience 15 --train_batch_size 128 --test_batch_size 256 \
        --lr_init 1e-3 --eval_every 3 $BASE $TATTN $CHB_INPUT"
}

# --- point-wise detection (smoothness lambda=0.3) ---
CFG_pointwise_tusz60 () {
  echo "--dataset TUSZ --task detection --model_name light_dense_hyper2 --dense_labels \
        --num_nodes 19 --max_seq_len 60 --time_step_size 1 \
        --rnn_units 64 --n_hyperedges 3 --tattn_n_layers 1 --smooth_weight 0.3 \
        --num_epochs 40 --patience 5 --train_batch_size 16 --test_batch_size 32 \
        --lr_init 3e-4 --eval_every 3 $BASE $TATTN $TUSZ_INPUT"
}
CFG_pointwise_chb () {
  echo "--dataset CHBMIT --task detection --model_name light_dense_hyper2 --dense_labels \
        --num_nodes 22 --max_seq_len 12 --time_step_size 1 \
        --rnn_units 64 --n_hyperedges 3 --tattn_n_layers 1 --smooth_weight 0.3 \
        --num_epochs 40 --patience 5 --train_batch_size 32 --test_batch_size 256 \
        --lr_init 3e-4 --eval_every 3 $BASE $TATTN $CHB_INPUT"
}

# --- seizure prediction (preictal vs interictal) ---
CFG_prediction_tusz12_K2 () {
  echo "--dataset TUSZ --task prediction --model_name light_st_hyper \
        --num_nodes 19 --max_seq_len 12 --time_step_size 1 --no_bidirectional \
        --rnn_units 128 --n_hyperedges 2 --aux_type none \
        --num_epochs 40 --patience 8 --train_batch_size 64 --test_batch_size 256 \
        --lr_init 1e-3 --eval_every 1 $BASE $TUSZ_INPUT"
}
CFG_prediction_tusz60_K1 () {
  echo "--dataset TUSZ --task prediction --model_name light_st_hyper \
        --num_nodes 19 --max_seq_len 60 --time_step_size 1 --no_bidirectional \
        --rnn_units 128 --n_hyperedges 1 --aux_type none \
        --num_epochs 60 --patience 12 --train_batch_size 32 --test_batch_size 128 \
        --lr_init 1e-3 --eval_every 1 $BASE $TUSZ_INPUT"
}
