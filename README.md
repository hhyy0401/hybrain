# Seizure Detection & Prediction — LightSTHyper

EEG seizure modeling on TUSZ and CHB-MIT across three tasks: window-based
(clip-level) detection, point-wise (per-second) detection, and seizure
prediction (preictal vs. interictal). Built on top of
[Kotoge/EvoBrain](https://github.com/Kotoge/EvoBrain) (MIT license).

Ships **seven trained checkpoints** — one per reported result — that you can
evaluate directly or retrain from scratch. Each reproduces its number exactly;
the full table and verification log are in [`RESULTS.md`](RESULTS.md).

## Quickstart

```bash
# 1. Environment
conda create -n evobrain python=3.10 && conda activate evobrain
pip install -r requirements.txt        # CUDA-matched torch + mamba-ssm, see Notes

# 2. Point DATA_DIR at your data (layout below)
export DATA_DIR=/path/to/data
export PYTHON=python

# 3a. Evaluate the released checkpoints
bash reproduce.sh                       # all 7
bash reproduce.sh window_chb_K3         # one

# 3b. Or train from scratch
bash train.sh window_chb_K3                       # seed 123
SEEDS="123 456 789" bash train.sh window_chb_K3   # 3 seeds -> mean +/- std
```

Both scripts read flags from [`configs.sh`](configs.sh) — the single source of
truth — so training and evaluation always match.

## Released checkpoints

| Dir (`ckpts/…`) | Task | Dataset | Clip | K |
|---|---|---|---|---|
| `window_tusz12`        | window-based detection | TUSZ    | 12 s | 1 |
| `window_chb_K1`        | window-based detection | CHB-MIT | 12 s | 1 |
| `window_chb_K3`        | window-based detection | CHB-MIT | 12 s | 3 |
| `pointwise_tusz60`     | point-wise detection   | TUSZ    | 60 s | 3 |
| `pointwise_chb`        | point-wise detection   | CHB-MIT | 12 s | 3 |
| `prediction_tusz12_K2` | seizure prediction     | TUSZ    | 12 s | 2 |
| `prediction_tusz60_K1` | seizure prediction     | TUSZ    | 60 s | 1 |

Each dir holds `best.pth.tar` (seed 123) and `args.json` (full config; data
paths anonymized to `<DATA_DIR>`).

## Data

Set `DATA_DIR` to the parent of the preprocessed folders:

```
$DATA_DIR/tusz_resampled/      $DATA_DIR/tusz_v2.0.6/edf/      # TUSZ
$DATA_DIR/chb_mit_resampled/   $DATA_DIR/chb_mit/             # CHB-MIT
```

- **TUSZ v2.0.6** — official patient-level split, 12 s and 60 s clips, 19 channels.
- **CHB-MIT** — same-patient random 15 % test split, 22 channels.

If your raw EDF tree is elsewhere, override it: `export TUSZ_RAW=…` / `CHB_RAW=…`.
The prediction loader reads raw EDF on the fly, so this must be correct for
prediction; window / point-wise mainly use the resampled `.h5`.

## Repo layout

```
src/
├── main.py                  # train / eval entry
├── args.py                  # CLI
├── utils.py  constants.py
├── model/                   # LightSTHyper (6 files)
│   ├── light_dyn_hyper.py        # window / prediction
│   ├── light_dyn_hyper2.py       # + temporal attention
│   ├── light_dense_hyper.py
│   ├── light_dense_hyper2.py     # point-wise
│   ├── mamba_backbone.py
│   └── temporal_backbones.py
└── data/                    # dataloaders, preprocessing, file_markers
ckpts/                       # 7 checkpoints + args.json
configs.sh  reproduce.sh  train.sh
RESULTS.md  requirements.txt
```

## Notes

- `torch` must be a CUDA build matching your driver (install from pytorch.org);
  `mamba-ssm` needs a CUDA toolchain. Tested with torch 2.5.1+cu121 on an A6000.
- `configs.sh` sets `--num_workers 0` (the prediction loader is not
  multiprocessing-safe).
- Reporting: test AUROC at dev-AUROC-best; F1 at dev max-F1 threshold τ\*; seeds
  {123, 456, 789}.

## License

Code: MIT (inherited from EvoBrain).
