"""
Build file-marker text files for CHB-MIT seizure PREDICTION (pre-ictal vs
inter-ictal binary classification), matching the same naming/format used by
our TUSZ prediction loader and the existing CHB detection markers.

Algorithm (mirrors TUSZ prediction build_file_markers_prediction.py):
  - For each seizure with onset s:
      pre_end   = s - BUFFER_DUR
      pre_start = pre_end - PRE_DUR             (strict 1-min preictal, clip_len < PRE)
                  OR pre_end - 4*clip_len       (widened search, clip_len >= PRE)
    Keep clip indices whose [cs, ce] lies fully in [pre_start, pre_end] and does
    not overlap any OTHER seizure's exclusion zone [s2-BUFFER, e2+BUFFER].
    These become pre-ictal (label=1).
  - Inter-ictal candidates: all clip indices that don't overlap any
    [s-BUFFER, e+BUFFER] exclusion zone.

Per-patient split: same 70/15/15 random split of clip indices we use for
detection (build_file_markers_chb.py), with a fixed --seed.

Per-patient interictal sub-sampling: each patient's inter-ictal clip count
is capped at `interictal_ratio * (that patient's pre-ictal count)`. Applied
identically to train/dev/test.

Output:  data/file_markers_prediction_chb/{trainSet,devSet,testSet}_seq2seq_<L>s_{sz,nosz}.txt
"""
import argparse
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from constants_chb import CHB_PATIENTS, FREQUENCY


# Prediction task constants — same as TUSZ prediction
PRE_DUR = 60.0          # 1 minute pre-ictal
BUFFER_DUR = 300.0      # 5 minute buffer on each side


# -------- parsing helpers (copied/adapted from build_file_markers_chb.py) --------

def time_to_seconds(text: str) -> float:
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if not m:
        raise ValueError(f"Cannot parse seconds from: {text!r}")
    return float(m.group(1))


def parse_summary(summary_path: Path):
    """Return {edf_basename_no_ext: [(start_s, end_s), ...]}."""
    out = {}
    current = None
    pending_starts = {}

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("File Name:"):
                current = line.split(":", 1)[1].strip()
                if current.endswith(".edf"):
                    current = current[:-4]
                out[current] = []
                pending_starts = {}
                continue
            if current is None:
                continue
            m = re.match(r"Seizure(?:\s+(\d+))?\s+Start Time\s*:\s*(.+)$", line)
            if m:
                idx = m.group(1)
                pending_starts[idx] = time_to_seconds(m.group(2))
                continue
            m = re.match(r"Seizure(?:\s+(\d+))?\s+End Time\s*:\s*(.+)$", line)
            if m:
                idx = m.group(1)
                start = pending_starts.pop(idx, None)
                if start is None:
                    continue
                end = time_to_seconds(m.group(2))
                out[current].append((start, end))
    return out


def h5_duration_sec(h5_path: Path) -> float:
    with h5py.File(h5_path, "r") as f:
        sig = f["resampled_signal"]
        # signal stored as (N, T) at FREQUENCY
        return sig.shape[1] / FREQUENCY


def num_clips_from_h5(h5_path: Path, clip_len: float) -> int:
    return int(h5_duration_sec(h5_path) // clip_len)


def overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


# -------- core marker logic (mirrors TUSZ prediction) --------

def build_markers_for_recording(edf_base: str, h5_path: Path, seizures, clip_len: float):
    """Return (pre_lines, inter_lines) for one recording.
    Each line: <edf_base>.edf_<clip_idx>.h5,<label>
    """
    duration = h5_duration_sec(h5_path)
    pre_lines: list[str] = []
    inter_lines: list[str] = []
    h5_name = edf_base + ".edf"

    # --- Pre-ictal: same grid-aligned logic as TUSZ ---
    for s, _ in seizures:
        pre_end = s - BUFFER_DUR
        if clip_len < PRE_DUR:
            pre_start = pre_end - PRE_DUR
        else:
            pre_start = pre_end - 4 * clip_len
        if pre_start < 0 or pre_end > duration:
            continue
        # exclusion w/ other seizures
        ok = True
        for s2, e2 in seizures:
            if s2 == s:
                continue
            if overlaps(pre_start, pre_end, max(0.0, s2 - BUFFER_DUR), e2 + BUFFER_DUR):
                ok = False
                break
        if not ok:
            continue
        first_idx = math.ceil(pre_start / clip_len)
        last_idx = math.floor(pre_end / clip_len) - 1
        for ci in range(first_idx, last_idx + 1):
            cs = ci * clip_len
            ce = cs + clip_len
            if cs < pre_start or ce > pre_end:
                continue
            pre_lines.append(f"{h5_name}_{ci}.h5,1")

    # --- Inter-ictal: clips that don't overlap any exclusion zone ---
    excl = [(max(0.0, s - BUFFER_DUR), e + BUFFER_DUR) for (s, e) in seizures]
    n_clips = int(duration // clip_len)
    for ci in range(n_clips):
        cs = ci * clip_len
        ce = cs + clip_len
        bad = False
        for (xs, xe) in excl:
            if overlaps(cs, ce, xs, xe):
                bad = True
                break
        if not bad:
            inter_lines.append(f"{h5_name}_{ci}.h5,0")
    return pre_lines, inter_lines


# -------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True,
                    help="Path to CHB-MIT raw dir (with chbXX/ subfolders)")
    ap.add_argument("--input_dir", required=True,
                    help="Path to chb_mit_resampled (.h5 files)")
    ap.add_argument("--out_dir", default="file_markers_prediction_chb",
                    help="Output directory under data/")
    ap.add_argument("--clip_len", type=float, default=12)
    ap.add_argument("--seed", type=int, default=123,
                    help="RNG seed for per-patient 70/15/15 split")
    ap.add_argument("--interictal_ratio", type=float, default=5.0,
                    help="Per-patient cap: inter-ictal count <= ratio * pre-ictal count")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    input_dir = Path(args.input_dir)
    out_dir = Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    # Per-patient: collect all marker lines (pre + inter), then split 70/15/15
    # and apply ratio cap.
    splits = {"train": {"sz": [], "nosz": []},
              "dev":   {"sz": [], "nosz": []},
              "test":  {"sz": [], "nosz": []}}

    for pid in CHB_PATIENTS:
        pdir = raw_dir / pid
        if not pdir.exists():
            print(f"  ! skip {pid}: dir not found")
            continue
        summary = pdir / f"{pid}-summary.txt"
        if not summary.exists():
            print(f"  ! skip {pid}: no summary file")
            continue
        sz_map = parse_summary(summary)

        # Collect all marker lines for this patient, also grouped by EDF
        all_pre = []
        all_inter = []
        for edf_base, sz_windows in sz_map.items():
            h5_path = input_dir / f"{edf_base}.h5"
            if not h5_path.exists():
                continue
            try:
                pre, inter = build_markers_for_recording(
                    edf_base, h5_path, sz_windows, args.clip_len)
            except Exception as ex:
                print(f"  ! {edf_base}: {ex}", file=sys.stderr)
                continue
            all_pre.extend(pre)
            all_inter.extend(inter)

        # Per-patient 70/15/15 split (using rng)
        rng.shuffle(all_pre)
        rng.shuffle(all_inter)

        n_pre = len(all_pre)
        if n_pre == 0:
            print(f"  {pid}: 0 pre-ictal clips (no usable seizures) — skip patient")
            continue
        # Cap inter-ictal at ratio * n_pre BEFORE splitting (so train/dev/test
        # all draw from the same capped pool, matching TUSZ behaviour).
        cap = int(args.interictal_ratio * n_pre)
        if len(all_inter) > cap:
            all_inter = py_rng.sample(all_inter, cap)

        def split_70_15_15(lst):
            n = len(lst)
            n_tr = int(0.70 * n)
            n_dv = int(0.15 * n)
            return lst[:n_tr], lst[n_tr:n_tr + n_dv], lst[n_tr + n_dv:]

        pre_tr, pre_dv, pre_te = split_70_15_15(all_pre)
        in_tr, in_dv, in_te = split_70_15_15(all_inter)

        splits["train"]["sz"].extend(pre_tr)
        splits["train"]["nosz"].extend(in_tr)
        splits["dev"]["sz"].extend(pre_dv)
        splits["dev"]["nosz"].extend(in_dv)
        splits["test"]["sz"].extend(pre_te)
        splits["test"]["nosz"].extend(in_te)

        print(f"  {pid}: pre={n_pre}  inter(capped)={len(all_inter)}  "
              f"-> tr={len(pre_tr)}/{len(in_tr)}  "
              f"dv={len(pre_dv)}/{len(in_dv)}  "
              f"te={len(pre_te)}/{len(in_te)}")

    # Write files
    print()
    for split in ("train", "dev", "test"):
        tag = "trainSet" if split == "train" else ("devSet" if split == "dev" else "testSet")
        for key in ("sz", "nosz"):
            fn = out_dir / f"{tag}_seq2seq_{int(args.clip_len)}s_{key}.txt"
            with open(fn, "w") as f:
                for line in splits[split][key]:
                    f.write(line + "\n")
            print(f"wrote {fn} ({len(splits[split][key])} lines)")

    # Also copy means/stds from detection markers (re-use)
    src_means = Path(__file__).resolve().parent / "file_markers_chb"
    for key in (f"means_seq2seq_fft_{int(args.clip_len)}s_szdetect_single.pkl",
                f"stds_seq2seq_fft_{int(args.clip_len)}s_szdetect_single.pkl"):
        src = src_means / key
        dst = out_dir / key
        if src.exists() and not dst.exists():
            import shutil
            shutil.copyfile(src, dst)
            print(f"  copied normalization stats: {key}")


if __name__ == "__main__":
    main()
