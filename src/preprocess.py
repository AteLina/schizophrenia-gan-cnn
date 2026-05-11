"""
Full MRI preprocessing pipeline for schizophrenia classification.
Datasets: UCLA CNP (ds000030) + COBRE (ds000115)

Pipeline steps:
    1. Skull stripping          — HD-BET
    2. Registration             — ANTsPy (MNI152 standard space)
    3. Bias field correction    — ANTsPy N4
    4. ComBat harmonization     — neuroCombat (across scanner sites)
    5. Z-score normalization    — per scan
    6. Slice extraction         — middle 70 axial slices → 256x256 PNG

Usage:
    python preprocess.py \
        --ucla_dir   data/raw/ucla_cnp \
        --cobre_dir  data/raw/cobre \
        --out_dir    data/preprocessed \
        --slices_dir data/slices
"""

# Imports
import os
import argparse
import logging
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Imported inside functions so the script can be inspected without full env
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Constants
SLICE_AXIS = 2  # axial = z-axis
N_SLICES = 70  # number of middle slices to extract
OUT_SIZE = (256, 256)  # resize target
RANDOM_SEED = 42

# All datasets use unified UCLA diagnostic codes after cleaning.
# Cleaned participants files are named participants_{dataset}_cleaned.tsv
# with columns: id, diagnosis (values: SCHZ or CONTROL)
SCZ_LABEL = "SCHZ"
HC_LABEL = "CONTROL"

# Walks dataset directories and builds a subject table with subject_id, t1_path, label, site, dataset
def collect_subjects(ucla_dir: Path, cobre_dir: Path, nusdast_dir: Path = None) -> pd.DataFrame:
    records = []

    # Shared parsing logic for all three datasets
    def _parse_dataset(data_dir: Path, tsv_name: str, site: str, dataset: str):
        tsv_path = data_dir / tsv_name
        if not tsv_path.exists():
            log.warning(f"{tsv_name} not found in {data_dir} — skipping {site}")
            return

        df = pd.read_csv(tsv_path, sep="\t")

        # Accept 'id' or 'participant_id' as the subject ID column
        id_col = next(
            (c for c in df.columns if c.lower() in {"id", "participant_id"}), None
        )
        if id_col is None:
            log.warning(f"No id/participant_id column found in {tsv_name} — skipping {site}")
            return

        # Accept 'diagnosis' column
        diag_col = next((c for c in df.columns if c.lower() == "diagnosis"), None)
        if diag_col is None:
            log.warning(f"No diagnosis column found in {tsv_name} — skipping {site}")
            return

        found = 0
        for _, row in df.iterrows():
            subj = str(row[id_col]).strip()
            diag = str(row[diag_col]).strip()

            if diag == SCZ_LABEL:
                label = 1
            elif diag == HC_LABEL:
                label = 0
            else:
                log.debug(f"Skipping {site} {subj} with diagnosis '{diag}'")
                continue

            # Prefix NUSDAST IDs to avoid collisions with UCLA/COBRE IDs
            subj_key = f"nusdast_{subj}" if dataset == "nusdast" else subj

            # Find T1 file — try BIDS layout, then common XNAT/flat layouts
            t1_path = data_dir / subj / "anat" / f"{subj}_T1w.nii.gz"
            if not t1_path.exists():
                t1_path = data_dir / subj / "anat" / f"{subj}_T1w.nii"
            if not t1_path.exists():
                # NUSDAST XNAT flat layout fallback
                candidates = list((data_dir / subj).rglob("*T1*.nii*")) if (data_dir / subj).exists() else []
                if candidates:
                    t1_path = candidates[0]
            if not t1_path.exists():
                log.debug(f"T1 not found for {site} {subj}, skipping")
                continue

            records.append({
                "subject_id": subj_key,
                "t1_path": str(t1_path),
                "label": label,
                "site": site,
                "dataset": dataset,
            })
            found += 1

        log.info(f"  {site}: loaded {found} subjects from {tsv_name}")

    # Parse all three datasets
    _parse_dataset(ucla_dir, "participants_ucla_cleaned.tsv", "UCLA", "ucla_cnp")
    _parse_dataset(cobre_dir, "participants_cobre_cleaned.tsv", "COBRE", "cobre")
    if nusdast_dir is not None and nusdast_dir.exists():
        _parse_dataset(nusdast_dir, "participants_nusdast_cleaned.tsv", "NUSDAST", "nusdast")
    elif nusdast_dir is not None:
        log.warning(f"NUSDAST directory not found at {nusdast_dir} — skipping")

    df = pd.DataFrame(records)
    log.info(f"Found {len(df)} subjects total "
             f"({(df.label == 1).sum()} SCZ, {(df.label == 0).sum()} HC)")
    log.info(f"  UCLA:    {(df.dataset == 'ucla_cnp').sum()} subjects")
    log.info(f"  COBRE:   {(df.dataset == 'cobre').sum()} subjects")
    log.info(f"  NUSDAST: {(df.dataset == 'nusdast').sum()} subjects")
    return df
    
# Skull stripping

# Runs HD-BET skull stripping and returns path to the brain-extracted file
def skull_strip(t1_path: Path, out_path: Path) -> Path:
    if out_path.exists():
        return out_path

    try:
        from HD_BET.run import run_hd_bet
    except ImportError:
        raise ImportError("HD-BET not installed. Run: pip install hd-bet")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_hd_bet(
        str(t1_path),
        str(out_path), # KEEP .nii.gz
        mode="fast",
        device="cpu",
        do_tta=False,
        bet=True,
        postprocess=True,
        keep_mask=False,
        overwrite=True,
    )

    return out_path

# Registration to MNI152

# Registers the skull-stripped scan to MNI152 standard space using ANTsPy affine transform
def register_to_mni(stripped_path: Path, out_path: Path, mni_template: Path) -> Path:
    if out_path.exists():
        return out_path

    try:
        import ants
    except ImportError:
        raise ImportError("ANTsPy not installed. Run: pip install antspyx")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixed = ants.image_read(str(mni_template))
    moving = ants.image_read(str(stripped_path))

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Affine", # linear is faster, appropriate for structural MRI
    )

    ants.image_write(reg["warpedmovout"], str(out_path))
    return out_path

# Bias field correction

# Runs N4 bias field correction using ANTsPy on the registered scan
def correct_bias_field(registered_path: Path, out_path: Path) -> Path:
    if out_path.exists():
        return out_path

    try:
        import ants
    except ImportError:
        raise ImportError("ANTsPy not installed. Run: pip install antspyx")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = ants.image_read(str(registered_path))
    corrected = ants.n4_bias_field_correction(img)
    ants.image_write(corrected, str(out_path))
    return out_path

# ComBat harmonization

# Runs ComBat across scanner sites and returns a dict of subject_id, harmonized numpy volume
def combat_harmonize(subjects_df: pd.DataFrame, preprocessed_dir: Path) -> dict:
    try:
        from neuroCombat import neuroCombat
    except ImportError:
        raise ImportError("neuroCombat not installed. Run: pip install neuroCombat")

    log.info("Loading volumes for ComBat harmonization...")
    volumes = {}
    valid_ids = []

    for _, row in tqdm(subjects_df.iterrows(), total=len(subjects_df)):
        subj = row["subject_id"]
        bias_path = preprocessed_dir / subj / "bias_corrected.nii.gz"
        if not bias_path.exists():
            log.warning(f"Bias-corrected file missing for {subj}, skipping ComBat")
            continue
        img = nib.load(str(bias_path))
        arr = img.get_fdata(dtype=np.float32)
        volumes[subj] = arr
        valid_ids.append(subj)

    if not valid_ids:
        raise RuntimeError("No valid volumes found for ComBat harmonization.")

    # Build feature matrix: mean intensity per axial slice (n_slices x n_subjects)
    all_arrs = [volumes[s] for s in valid_ids]
    n_slices_z = all_arrs[0].shape[2]
    feature_matrix = np.array([
        [arr[:, :, z].mean() for z in range(n_slices_z)]
        for arr in all_arrs
    ]).T # shape: (n_slices_z, n_subjects)

    # Build covariates
    subj_df = subjects_df[subjects_df["subject_id"].isin(valid_ids)].set_index("subject_id")
    subj_df = subj_df.loc[valid_ids] # preserve order
    batch = subj_df["site"].values
    covars_df = pd.DataFrame({
        "diagnosis": subj_df["label"].values,
    })

    log.info(f"Running ComBat on {len(valid_ids)} subjects across sites: {np.unique(batch)}")
    combat_out = neuroCombat(
        dat=feature_matrix,
        covars=covars_df,
        batch_col="diagnosis", # preserve diagnosis effect
        categorical_cols=["diagnosis"],
    )

    # Scale each volume by the ratio of harmonized to original mean per slice
    harmonized_volumes = {}
    for i, subj in enumerate(valid_ids):
        orig_means = feature_matrix[:, i] # (n_slices_z,)
        combat_means = combat_out["data"][:, i] # (n_slices_z,)
        # Avoid division by zero
        ratio = np.where(orig_means != 0, combat_means / orig_means, 1.0)
        arr = volumes[subj].copy()
        for z in range(n_slices_z):
            arr[:, :, z] *= ratio[z]
        harmonized_volumes[subj] = arr

    log.info("ComBat harmonization complete.")
    return harmonized_volumes

# Z-score normalization

# Z-score normalizes a 3D volume using only brain voxels (non-zero mask)
def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    mask = volume > 0
    mean = volume[mask].mean()
    std = volume[mask].std()
    if std == 0:
        return volume - mean
    normalized = volume.copy()
    normalized[mask] = (volume[mask] - mean) / std
    return normalized

# Slice extraction

# Extracts the middle n_slices axial slices from a volume and saves them as 256x256 PNGs
def extract_slices(
        volume: np.ndarray,
        subject_id: str,
        label: int,
        slices_dir: Path,
        n_slices: int = N_SLICES,
        out_size: tuple = OUT_SIZE,
) -> list:
    class_name = "schizophrenia" if label == 1 else "healthy"
    out_dir = slices_dir / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_z = volume.shape[2]
    center = n_z // 2
    start = max(0, center - n_slices // 2)
    end = min(n_z, start + n_slices)
    slice_idxs = range(start, end)

    saved = []
    for z in slice_idxs:
        sl = volume[:, :, z]

        # Clip to 1st–99th percentile to remove outlier voxels before scaling
        p1, p99 = np.percentile(sl, [1, 99])
        sl = np.clip(sl, p1, p99)

        # Scale to 0–255
        sl_min, sl_max = sl.min(), sl.max()
        if sl_max > sl_min:
            sl = (sl - sl_min) / (sl_max - sl_min) * 255.0
        else:
            sl = np.zeros_like(sl)

        img = Image.fromarray(sl.astype(np.uint8))
        img = img.resize(out_size, Image.LANCZOS)
        fname = f"{subject_id}_slice{z:04d}.png"
        fpath = out_dir / fname
        img.save(str(fpath))
        saved.append(str(fpath))

    return saved

# Download MNI template

# Downloads MNI152 1mm template if not already present
def get_mni_template(template_dir: Path) -> Path:
    import urllib.request
    template_path = template_dir / "MNI152_T1_1mm_brain.nii.gz"
    if template_path.exists():
        return template_path

    template_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://github.com/ANTsX/ANTs/raw/master/Examples/Data/"
        "MNI152_T1_1mm_brain.nii.gz"
    )
    log.info("Downloading MNI152 template...")
    urllib.request.urlretrieve(url, str(template_path))
    log.info(f"MNI152 template saved to {template_path}")
    return template_path

# Main pipeline

# Runs the full preprocessing pipeline for all subjects
def run_pipeline(args):
    ucla_dir = Path(args.ucla_dir).expanduser()
    cobre_dir = Path(args.cobre_dir).expanduser()
    nusdast_dir = Path(args.nusdast_dir).expanduser() if args.nusdast_dir else None
    out_dir = Path(args.out_dir).expanduser()
    slices_dir = Path(args.slices_dir).expanduser()
    template_dir = out_dir / "templates"

    out_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)

    # Collect subjects
    subjects_df = collect_subjects(ucla_dir, cobre_dir, nusdast_dir)
    if args.test_run:
        subjects_df = subjects_df.sample(n=min(10, len(subjects_df)),
                                         random_state=RANDOM_SEED)
        log.info(f"TEST RUN: processing {len(subjects_df)} subjects only")

    subjects_df.to_csv(out_dir / "subjects.csv", index=False)
    log.info(f"Subject manifest saved to {out_dir / 'subjects.csv'}")

    # Get MNI template
    mni_template = get_mni_template(template_dir)

    # Per-subject preprocessing
    log.info("Starting per-subject preprocessing (skull strip → register → bias correct)...")
    failed = []

    for _, row in tqdm(subjects_df.iterrows(), total=len(subjects_df), desc="Preprocessing"):
        subj = row["subject_id"]
        t1_path = Path(row["t1_path"])
        subj_dir = out_dir / subj
        subj_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Skull stripping
            stripped_path = subj_dir / "skull_stripped.nii.gz"
            skull_strip(t1_path, stripped_path)

            # Registration to MNI152
            registered_path = subj_dir / "registered.nii.gz"
            register_to_mni(stripped_path, registered_path, mni_template)

            # Bias field correction
            bias_path = subj_dir / "bias_corrected.nii.gz"
            correct_bias_field(registered_path, bias_path)

        except Exception as e:
            log.error(f"Failed preprocessing for {subj}: {e}")
            failed.append(subj)
            continue

    if failed:
        log.warning(f"{len(failed)} subjects failed preprocessing: {failed}")
        subjects_df = subjects_df[~subjects_df["subject_id"].isin(failed)]

    # ComBat harmonization
    log.info("Running ComBat harmonization...")
    harmonized_volumes = combat_harmonize(subjects_df, out_dir)

    # Normalize and extract slices
    log.info("Normalizing and extracting slices...")
    slice_manifest = []

    for _, row in tqdm(subjects_df.iterrows(), total=len(subjects_df), desc="Slicing"):
        subj = row["subject_id"]
        label = row["label"]

        if subj not in harmonized_volumes:
            log.warning(f"No harmonized volume for {subj}, skipping slice extraction")
            continue

        # Z-score normalize
        volume = zscore_normalize(harmonized_volumes[subj])

        # Extract slices → PNG
        saved_paths = extract_slices(volume, subj, label, slices_dir)
        for p in saved_paths:
            slice_manifest.append({
                "subject_id": subj,
                "label": label,
                "site": row["site"],
                "dataset": row["dataset"],
                "slice_path": p,
            })

    # Save slice manifest
    manifest_df = pd.DataFrame(slice_manifest)
    manifest_df.to_csv(slices_dir / "slice_manifest.csv", index=False)
    log.info(f"Saved {len(manifest_df)} slices total.")
    log.info(f"  SCZ slices: {(manifest_df.label == 1).sum()}")
    log.info(f"  HC  slices: {(manifest_df.label == 0).sum()}")
    log.info(f"Slice manifest saved to {slices_dir / 'slice_manifest.csv'}")
    log.info("Preprocessing complete.")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI Preprocessing Pipeline")
    parser.add_argument("--ucla_dir", required=True, help="Path to UCLA CNP raw data")
    parser.add_argument("--cobre_dir", required=True, help="Path to COBRE raw data")
    parser.add_argument("--nusdast_dir", required=False, default=None,
                        help="Path to NUSDAST raw data (optional — omit if not yet downloaded)")
    parser.add_argument("--out_dir", required=True, help="Path for preprocessed NIfTI output")
    parser.add_argument("--slices_dir", required=True, help="Path for PNG slice output")
    parser.add_argument("--test_run", action="store_true",
                        help="Process only 10 subjects (for pipeline testing)")
    args = parser.parse_args()
    run_pipeline(args)
