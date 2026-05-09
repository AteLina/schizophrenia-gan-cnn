"""
make_slices.py
==============
Standalone script to convert already-preprocessed bias_corrected.nii.gz files
 into 256x256 PNG slices

Use this when preprocessing was interrupted partway through — it works on
whatever subjects are already done.

Pipeline:
    1. Scan preprocessed_dir for subjects with bias_corrected.nii.gz
    2. Load subjects.csv to get diagnosis labels
    3. ComBat harmonization across completed subjects
    4. Z-score normalization per scan
    5. Extract middle 70 axial slices → 256x256 PNG

Usage:
    python make_slices.py \
        --preprocessed_dir /content/drive/MyDrive/schizophrenia_gan/data/preprocessed \
        --slices_dir       /content/drive/MyDrive/schizophrenia_gan/data/slices

Requirements:
    pip install neuroCombat nibabel numpy Pillow tqdm pandas
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

N_SLICES    = 70
OUT_SIZE    = (256, 256)
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FIND COMPLETED SUBJECTS
# ─────────────────────────────────────────────────────────────────────────────

def find_completed_subjects(preprocessed_dir: Path, subjects_csv: Path) -> pd.DataFrame:
    """
    Scan preprocessed_dir for subjects that have bias_corrected.nii.gz,
    then join with subjects.csv to get labels and site info.
    """
    # Load the subject manifest written by preprocess.py
    if not subjects_csv.exists():
        raise FileNotFoundError(
            f"subjects.csv not found at {subjects_csv}. "
            "Make sure you're pointing at the preprocessed output directory."
        )

    subjects_df = pd.read_csv(subjects_csv)

    # Find which subjects actually have a completed bias_corrected.nii.gz
    completed = []
    for _, row in subjects_df.iterrows():
        subj      = row["subject_id"]
        bias_path = preprocessed_dir / subj / "bias_corrected.nii.gz"
        if bias_path.exists():
            completed.append(subj)

    completed_df = subjects_df[subjects_df["subject_id"].isin(completed)].copy()
    skipped      = len(subjects_df) - len(completed_df)

    log.info(f"Found {len(completed_df)} completed subjects out of {len(subjects_df)} total")
    log.info(f"  SCZ: {(completed_df.label==1).sum()}")
    log.info(f"  HC:  {(completed_df.label==0).sum()}")
    log.info(f"  Skipped (not yet preprocessed): {skipped}")
    log.info(f"  Sites: {completed_df['site'].value_counts().to_dict()}")

    return completed_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — COMBAT HARMONIZATION
# ─────────────────────────────────────────────────────────────────────────────

def combat_harmonize(subjects_df: pd.DataFrame, preprocessed_dir: Path) -> dict:
    """
    Run ComBat harmonization across scanner sites on completed subjects only.
    Returns dict: subject_id -> harmonized numpy volume array.
    """
    try:
        from neuroCombat import neuroCombat
    except ImportError:
        raise ImportError("neuroCombat not installed. Run: pip install neuroCombat")

    log.info("Loading volumes for ComBat harmonization...")
    volumes   = {}
    valid_ids = []

    for _, row in tqdm(subjects_df.iterrows(), total=len(subjects_df), desc="Loading"):
        subj      = row["subject_id"]
        bias_path = preprocessed_dir / subj / "bias_corrected.nii.gz"
        if not bias_path.exists():
            continue
        img = nib.load(str(bias_path))
        arr = img.get_fdata(dtype=np.float32)
        volumes[subj] = arr
        valid_ids.append(subj)

    if not valid_ids:
        raise RuntimeError("No valid volumes found.")

    # Check if we have more than one site — ComBat requires at least 2 batches
    subj_df  = subjects_df[subjects_df["subject_id"].isin(valid_ids)].set_index("subject_id")
    subj_df  = subj_df.loc[valid_ids]
    sites    = subj_df["site"].values
    n_sites  = len(np.unique(sites))

    if n_sites < 2:
        log.warning(
            f"Only 1 site found ({np.unique(sites)[0]}) — "
            "ComBat requires 2+ sites. Skipping harmonization, returning raw volumes."
        )
        return volumes

    # Build feature matrix: mean intensity per axial slice
    all_arrs   = [volumes[s] for s in valid_ids]
    n_slices_z = all_arrs[0].shape[2]
    feature_matrix = np.array([
        [arr[:, :, z].mean() for z in range(n_slices_z)]
        for arr in all_arrs
    ]).T  # shape: (n_slices_z, n_subjects)

    covars_df = pd.DataFrame({
        "diagnosis": subj_df["label"].values,
        "site":      sites,
    })

    log.info(f"Running ComBat on {len(valid_ids)} subjects across {n_sites} sites: {np.unique(sites)}")

    try:
        combat_out = neuroCombat(
            dat=feature_matrix,
            covars=covars_df,
            batch_col="site",
            categorical_cols=["diagnosis"],
        )
    except Exception as e:
        log.warning(f"ComBat failed ({e}) — returning raw volumes without harmonization.")
        return volumes

    # Scale each volume by harmonized/original ratio per slice
    harmonized = {}
    for i, subj in enumerate(valid_ids):
        orig_means   = feature_matrix[:, i]
        combat_means = combat_out["data"][:, i]
        ratio        = np.where(orig_means != 0, combat_means / orig_means, 1.0)
        arr          = volumes[subj].copy()
        for z in range(n_slices_z):
            arr[:, :, z] *= ratio[z]
        harmonized[subj] = arr

    log.info("ComBat harmonization complete.")
    return harmonized


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Z-SCORE NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize using brain voxels only (non-zero mask)."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std  = volume[mask].std()
    if std == 0:
        return volume - mean
    normalized       = volume.copy()
    normalized[mask] = (volume[mask] - mean) / std
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SLICE EXTRACTION → PNG
# ─────────────────────────────────────────────────────────────────────────────

def extract_slices(
    volume: np.ndarray,
    subject_id: str,
    label: int,
    slices_dir: Path,
    n_slices: int = N_SLICES,
    out_size: tuple = OUT_SIZE,
) -> list:
    """
    Extract the middle n_slices axial slices → 256x256 PNG.
    Saves to slices_dir/schizophrenia/ or slices_dir/healthy/.
    """
    class_name = "schizophrenia" if label == 1 else "healthy"
    out_dir    = slices_dir / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_z        = volume.shape[2]
    center     = n_z // 2
    start      = max(0, center - n_slices // 2)
    end        = min(n_z, start + n_slices)

    saved = []
    for z in range(start, end):
        sl = volume[:, :, z]

        # Clip to 1st-99th percentile then scale to 0-255
        p1, p99 = np.percentile(sl, [1, 99])
        sl      = np.clip(sl, p1, p99)
        sl_min, sl_max = sl.min(), sl.max()
        if sl_max > sl_min:
            sl = (sl - sl_min) / (sl_max - sl_min) * 255.0
        else:
            sl = np.zeros_like(sl)

        img   = Image.fromarray(sl.astype(np.uint8))
        img   = img.resize(out_size, Image.LANCZOS)
        fname = f"{subject_id}_slice{z:04d}.png"
        fpath = out_dir / fname
        img.save(str(fpath))
        saved.append(str(fpath))

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    preprocessed_dir = Path(args.preprocessed_dir).expanduser()
    slices_dir       = Path(args.slices_dir).expanduser()
    subjects_csv     = preprocessed_dir / "subjects.csv"

    slices_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find completed subjects
    subjects_df = find_completed_subjects(preprocessed_dir, subjects_csv)

    if len(subjects_df) == 0:
        log.error("No completed subjects found. Check your preprocessed_dir path.")
        return

    # Step 2: ComBat harmonization
    log.info("Running ComBat harmonization...")
    harmonized_volumes = combat_harmonize(subjects_df, preprocessed_dir)

    # Steps 3-4: Normalize and extract slices
    log.info("Normalizing and extracting slices...")
    slice_manifest = []

    for _, row in tqdm(subjects_df.iterrows(), total=len(subjects_df), desc="Slicing"):
        subj  = row["subject_id"]
        label = row["label"]

        if subj not in harmonized_volumes:
            log.warning(f"No volume for {subj}, skipping")
            continue

        volume      = zscore_normalize(harmonized_volumes[subj])
        saved_paths = extract_slices(volume, subj, label, slices_dir)

        for p in saved_paths:
            slice_manifest.append({
                "subject_id": subj,
                "label":      label,
                "site":       row["site"],
                "dataset":    row["dataset"],
                "slice_path": p,
            })

    # Save manifest
    manifest_df = pd.DataFrame(slice_manifest)
    manifest_df.to_csv(slices_dir / "slice_manifest.csv", index=False)

    log.info(f"Done. Saved {len(manifest_df)} slices from {len(subjects_df)} subjects.")
    log.info(f"  SCZ slices: {(manifest_df.label==1).sum()}")
    log.info(f"  HC  slices: {(manifest_df.label==0).sum()}")
    log.info(f"  Slice manifest: {slices_dir / 'slice_manifest.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert already-preprocessed NIfTI files to PNG slices."
    )
    parser.add_argument(
        "--preprocessed_dir", required=True,
        help="Path to preprocessed output dir (must contain subjects.csv)"
    )
    parser.add_argument(
        "--slices_dir", required=True,
        help="Path to save PNG slices"
    )
    args = parser.parse_args()
    run(args)