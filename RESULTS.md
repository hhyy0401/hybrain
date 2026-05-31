# Results — released best models

Seven checkpoints under [`ckpts/`](ckpts/), each verified to reproduce its
number from the frozen weights (see [Verification](#verification)). Train from
scratch with [`train.sh`](train.sh) or evaluate with [`reproduce.sh`](reproduce.sh).

Numbers are **mean ± std over seeds {123, 456, 789}**. Test AUROC at the
dev-AUROC-best checkpoint; F1 at the dev max-F1 threshold τ\*. `K` = number of
hyperedges (`--n_hyperedges`).

## Window-based (clip-level) detection

| Setting | TUSZ 12 s AUROC / F1 | CHB-MIT 12 s AUROC / F1 | Checkpoint |
|---|---|---|---|
| Ours (K=1) | 0.899 ± 0.006 / 0.551 ± 0.008 | 0.917 ± 0.005 / 0.182 ± 0.033 | `window_tusz12`, `window_chb_K1` |
| Ours (K=3) | — | 0.924 ± 0.003 / 0.191 ± 0.009 | `window_chb_K3` |

## Point-wise (per-second) detection — smoothness λ=0.3

Frame-level AUROC / F1.

| Setting | TUSZ 60 s AUROC / F1 | CHB-MIT 12 s AUROC / F1 | Checkpoint |
|---|---|---|---|
| Ours (λ=0.3) | 0.938 ± 0.004 / 0.575 ± 0.049 | 0.929 ± 0.006 / 0.232 ± 0.015 | `pointwise_tusz60`, `pointwise_chb` |

## Seizure prediction (preictal vs. interictal, TUSZ)

| Setting | TUSZ 12 s AUROC / F1 | TUSZ 60 s AUROC / F1 | Checkpoint |
|---|---|---|---|
| Ours | 0.560 ± 0.031 / 0.278 ± 0.011 (K=2) | 0.758 ± 0.028 / 0.434 ± 0.037 (K=1) | `prediction_tusz12_K2`, `prediction_tusz60_K1` |

## Checkpoint index

| Dir | Task | Dataset | Clip | K |
|---|---|---|---|---|
| `window_tusz12`        | window     | TUSZ    | 12 s | 1 |
| `window_chb_K1`        | window     | CHB-MIT | 12 s | 1 |
| `window_chb_K3`        | window     | CHB-MIT | 12 s | 3 |
| `pointwise_tusz60`     | point-wise | TUSZ    | 60 s | 3 |
| `pointwise_chb`        | point-wise | CHB-MIT | 12 s | 3 |
| `prediction_tusz12_K2` | prediction | TUSZ    | 12 s | 2 |
| `prediction_tusz60_K1` | prediction | TUSZ    | 60 s | 1 |

Each dir holds `best.pth.tar` (seed 123) and `args.json` (full config; data
paths anonymized to `<DATA_DIR>`).

## Verification

Each checkpoint was reloaded with `--test --load_model_path` and re-evaluated;
the recomputed metrics and τ\* match the training logs exactly (seed 123):

| Checkpoint | Metric | Original | Reproduced |
|---|---|---|---|
| window_tusz12        | AUROC / F1       | 0.895 / 0.547 | 0.895 / 0.547 ✅ |
| window_chb_K1        | AUROC / F1       | 0.923 / 0.220 | 0.923 / 0.220 ✅ |
| window_chb_K3        | AUROC / F1       | 0.926 / 0.181 | 0.926 / 0.181 ✅ |
| pointwise_tusz60     | frame AUROC / F1 | 0.943 / 0.621 | 0.943 / 0.621 ✅ |
| pointwise_chb        | frame AUROC / F1 | 0.935 / 0.218 | 0.935 / 0.218 ✅ |
| prediction_tusz12_K2 | AUROC / F1       | 0.527 / 0.266 | 0.527 / 0.266 ✅ |
| prediction_tusz60_K1 | AUROC / F1       | 0.785 / 0.397 | 0.785 / 0.397 ✅ |

> CHB-MIT uses 22 channels (`--num_nodes 22`); TUSZ uses 19. Per-checkpoint
> flags live in each `ckpts/<name>/args.json`.
