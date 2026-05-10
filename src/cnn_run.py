"""
ResNet-18 CNN for schizophrenia vs. control classification.

Conditions tested (runs 8 tests total):
  1. No augmentation
  2. Traditional augmentation (on-the-fly)
  3. DCGAN synthetic augmentation  (25%, 50%, 100% ratio)
  4. PatchGAN synthetic augmentation (25%, 50%, 100% ratio)

Usage:
    python cnn_run.py \
        --real-schiz-dir data/real/schizophrenia \
        --real-ctrl-dir  data/real/control \
        --dcgan-dir      data/augmented/dcgan/schizophrenia \
        --patchgan-dir   data/augmented/patchgan/schizophrenia \
        --out-dir        results
"""

# Imports
import re
import sys
import json
import random
import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    balanced_accuracy_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms, models
from PIL import Image

warnings.filterwarnings("ignore")

# Constants
LABEL_SCHIZ = 1
LABEL_CTRL = 0
IMG_SIZE = 224 # ResNet-18 expects 224×224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fixes random seed
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Loads PNG brain MRI slices from two class directories
# Transform is applied lazily in __getitem__ so it can be swapped per-condition
class SliceDataset(Dataset):
    def __init__(self, schiz_dir: Path, ctrl_dir: Path, transform=None):
        self.transform = transform
        self.samples: list = []                       # list of (Path, int)

        for path in sorted(schiz_dir.glob("*.png")):
            self.samples.append((path, LABEL_SCHIZ))
        for path in sorted(ctrl_dir.glob("*.png")):
            self.samples.append((path, LABEL_CTRL))

        if not self.samples:
            raise RuntimeError(
                f"No PNG files found in {schiz_dir} or {ctrl_dir}. "
                "Check your --real-schiz-dir and --real-ctrl-dir paths."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

    @property
    def labels(self) -> list:
        return [lbl for _, lbl in self.samples]

    @property
    def paths(self) -> list:
        return [p for p, _ in self.samples]

# Pre-generated synthetic PNG images all labeled as schizophrenia (1)
class SyntheticDataset(Dataset):
    def __init__(self, syn_dir: Path, transform=None, n_samples: Optional[int] = None):
        paths = sorted(syn_dir.glob("*.png"))
        if not paths:
            raise RuntimeError(f"No PNG files found in {syn_dir}.")
        if n_samples is not None:
            paths = paths[:n_samples]
        self.samples = [(p, LABEL_SCHIZ) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# Wraps a subset of SliceDataset with a given transform
class _TransformedSubset(Dataset):
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# Applies transform to subset of the dataset
def subset_with_transform(dataset: SliceDataset, indices: list, transform) -> Dataset:
    return _TransformedSubset(Subset(dataset, indices), transform)

# Resizing and normalization, used for validation, test, and no-augmentation training
def base_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# On-the-fly augmentation for the traditional augmentation condition
def traditional_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# Parses subject ID from filenames to split by subject
def _extract_subject_id(path: Path) -> str:
    """
    Handles common patterns:
      sub-001_slice_042.png      →  sub-001
      UCLA_schiz_0043_slice_12.png →  UCLA_schiz_0043
    Falls back to the full stem if no pattern matches (one subject per file).
    """
    name = path.stem
    m = re.match(r"(sub-[^_]+)", name) # BIDS-style
    if m:
        return m.group(1)
    m = re.match(r"(.+?)_slice_\d+", name) # PREFIX_slice_N
    if m:
        return m.group(1)
    return name

# Split by subject and not slice to prevent data leakage and keeps class balance in each split
# Returns (train_indices, val_indices, test_indices)
def subject_split(
    dataset: SliceDataset,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    seed: int         = 42,
) -> tuple:
    rng = random.Random(seed)
    groups: dict = defaultdict(list)

    for i, (path, label) in enumerate(dataset.samples):
        sid = _extract_subject_id(path)
        groups[(sid, label)].append(i)

    schiz_subjects = [k for k in groups if k[1] == LABEL_SCHIZ]
    ctrl_subjects  = [k for k in groups if k[1] == LABEL_CTRL]

    def split_list(items):
        items = list(items)
        rng.shuffle(items)
        n       = len(items)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    tr_s, va_s, te_s = split_list(schiz_subjects)
    tr_c, va_c, te_c = split_list(ctrl_subjects)

    def flatten(keys):
        return [i for k in keys for i in groups[k]]

    return (
        flatten(tr_s + tr_c),
        flatten(va_s + va_c),
        flatten(te_s + te_c),
    )

# Downloads ResNet-18 pretrained on ImageNet, replaces final connected layer with single logit for binary classification
# BCEWithLogitsLoss (Binary Cross Entropy) is applied during training (numerically stable)
def build_resnet18() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# Runs one full training lap, feeds batches through models, measures error, updates weights
def _train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# Check validation set without changing
@torch.no_grad()
def _eval_epoch(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        preds   = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n

# Full training loop with early stopping, saves best checkpoint
def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    epochs: int,
    lr: float,
    patience: int,
    save_path: Path,
) -> dict:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss    = float("inf")
    patience_counter = 0
    history          = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss         = _train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss    = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return history

# Returns (y_true, y_prob) numpy arrays over the full loader
@torch.no_grad()
def predict(model, loader: DataLoader, device) -> tuple:
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device)).squeeze(1)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)

# Takes predictions and calculates accuracy, balanced accuracy, sensitivity, specificity, and AUC (Area Under the Curve)
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "balanced_acc":     balanced_accuracy_score(y_true, y_pred),
        "sensitivity":      tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity":      tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "auc":              roc_auc_score(y_true, y_prob),
        "confusion_matrix": cm,
        "y_true":           y_true,
        "y_prob":           y_prob,
    }

# Plotting Functions

def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Control", "Schizophrenia"],
        yticklabels=["Control", "Schizophrenia"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_roc_curves(results: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, m in results.items():
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_prob"])
        ax.plot(fpr, tpr, label=f"{label} (AUC={m['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Conditions")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

def plot_accuracy_bar(results: dict, save_path: Path) -> None:
    labels = list(results.keys())
    accs   = [results[l]["accuracy"] for l in labels]
    aucs   = [results[l]["auc"]      for l in labels]

    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy",  color="steelblue")
    ax.bar(x + width / 2, aucs, width, label="AUC",       color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy and AUC — All Conditions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

# Line graph: test performance vs. synthetic data proportion
def plot_proportion_lines(proportion_results: dict, save_path: Path) -> None:
    """
    proportion_results = { "dcgan": {0.25: metrics, 0.50: metrics, 1.00: metrics},
                            "patchgan": {...} }
    """
    ratios       = [0.25, 0.50, 1.00]
    ratio_labels = ["25%", "50%", "100%"]
    colors       = {"dcgan": "royalblue", "patchgan": "tomato"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for metric, ax in zip(["accuracy", "auc"], axes):
        for gan_type, ratio_dict in proportion_results.items():
            if not ratio_dict:
                continue
            values = [ratio_dict.get(r, {}).get(metric, None) for r in ratios]
            valid  = [(rl, v) for rl, v in zip(ratio_labels, values) if v is not None]
            if not valid:
                continue
            rl_vals, v_vals = zip(*valid)
            ax.plot(rl_vals, v_vals, marker="o",
                    label=gan_type.upper(), color=colors.get(gan_type))
        ax.set_title(metric.upper())
        ax.set_xlabel("Synthetic proportion (% of real schiz count)")
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0.4, 1.05)
        ax.legend()

    fig.suptitle("Performance vs. Synthetic Data Proportion", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

# Wraps datasets into loaders that feed images to the model in batches
def _make_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int) -> tuple:
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )

# Runs one full experiment
# Trains a fresh ResNet-18, evaluates on test set, saves confusion matrix and training curves, returns metrics
def run_condition(
    condition_name: str,
    train_ds,
    val_ds,
    test_ds,
    args,
    device,
    out_dir: Path,
) -> dict:
    tag = condition_name.replace(" ", "_")
    print(f"\n{'=' * 62}")
    print(f"  Condition : {condition_name}")
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    print(f"{'=' * 62}")

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, args.batch_size, args.num_workers
    )

    model = build_resnet18().to(device)
    history = train_model(
        model, train_loader, val_loader, device,
        epochs    = args.epochs,
        lr        = args.lr,
        patience  = args.patience,
        save_path = out_dir / f"{tag}_best.pt",
    )

    # Save training curves
    _plot_training_history(history, condition_name, out_dir / f"{tag}_history.png")

    y_true, y_prob = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_prob)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        title     = f"Confusion Matrix — {condition_name}",
        save_path = out_dir / f"{tag}_cm.png",
    )

    print(
        f"  Test  acc={metrics['accuracy']:.4f}  "
        f"bal_acc={metrics['balanced_acc']:.4f}  "
        f"AUC={metrics['auc']:.4f}  "
        f"sens={metrics['sensitivity']:.4f}  "
        f"spec={metrics['specificity']:.4f}"
    )
    return metrics

# Plots loss and validation accuracy curves over epochs and saves as PNG
def _plot_training_history(history: dict, title: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_acc"], color="green")
    axes[1].set_title("Val Accuracy")
    axes[1].set_xlabel("Epoch")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# Main

# Reads the folder paths, settings, and options typed in Terminal when running the script
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ResNet-18 schizophrenia classifier — GAN augmentation study"
    )
    # Data paths
    p.add_argument("--real-schiz-dir", type=Path, required=True,
                   help="Directory of real schizophrenia PNG slices")
    p.add_argument("--real-ctrl-dir",  type=Path, required=True,
                   help="Directory of real control PNG slices")
    p.add_argument("--dcgan-dir",      type=Path, default=None,
                   help="Directory of DCGAN synthetic schiz PNG slices (pool)")
    p.add_argument("--patchgan-dir",   type=Path, default=None,
                   help="Directory of PatchGAN synthetic schiz PNG slices (pool)")
    # Output
    p.add_argument("--out-dir",        type=Path, default=Path("results"),
                   help="Output directory for models, plots, and tables")
    # Training
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--patience",       type=int,   default=10,
                   help="Early stopping patience (epochs)")
    p.add_argument("--num-workers",    type=int,   default=4)
    # Experiment
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--train-frac",     type=float, default=0.70)
    p.add_argument("--val-frac",       type=float, default=0.15)
    p.add_argument("--conditions",     nargs="+",
                   default=["no_aug", "traditional", "dcgan", "patchgan"],
                   choices=["no_aug", "traditional", "dcgan", "patchgan"],
                   help="Which conditions to run")
    p.add_argument("--syn-ratios",     nargs="+", type=float,
                   default=[0.25, 0.50, 1.00],
                   help="Synthetic-to-real schiz ratios for GAN conditions")
    return p.parse_args()

def main() -> None:
    args = build_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Seed   : {args.seed}")
    print(f"Output : {args.out_dir}\n")

    # Load full real dataset
    full_ds = SliceDataset(
        schiz_dir=args.real_schiz_dir,
        ctrl_dir=args.real_ctrl_dir,
        transform=None, # transform applied per-condition below
    )
    n_schiz = sum(l == LABEL_SCHIZ for l in full_ds.labels)
    n_ctrl  = sum(l == LABEL_CTRL  for l in full_ds.labels)
    print(f"Total real slices: {len(full_ds)}  (schiz={n_schiz}, ctrl={n_ctrl})")

    # Subject-level split, test set locked until final evaluation
    train_idx, val_idx, test_idx = subject_split(
        full_ds, args.train_frac, args.val_frac, args.seed
    )
    print(f"Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    n_train_schiz = sum(full_ds.labels[i] == LABEL_SCHIZ for i in train_idx)
    print(f"Real schiz in training set: {n_train_schiz}\n")

    base_tf = base_transform()
    trad_tf = traditional_transform()

    # Test set is always base transform — never touched until predict()
    test_ds = subset_with_transform(full_ds, test_idx, base_tf)

    all_results: dict        = {}
    proportion_results: dict = {"dcgan": {}, "patchgan": {}}

    # Condition 1: No Augmentation
    if "no_aug" in args.conditions:
        all_results["No Aug"] = run_condition(
            "No Aug",
            train_ds = subset_with_transform(full_ds, train_idx, base_tf),
            val_ds   = subset_with_transform(full_ds, val_idx,   base_tf),
            test_ds  = test_ds,
            args     = args,
            device   = device,
            out_dir  = args.out_dir,
        )

    # Condition 2: Traditional Augmentation
    if "traditional" in args.conditions:
        all_results["Traditional"] = run_condition(
            "Traditional",
            train_ds = subset_with_transform(full_ds, train_idx, trad_tf),
            val_ds   = subset_with_transform(full_ds, val_idx,   base_tf),
            test_ds  = test_ds,
            args     = args,
            device   = device,
            out_dir  = args.out_dir,
        )

    # Conditions 3 & 4: GAN Augmentation (multiple mixing ratios)
    for gan_name, syn_dir in [("dcgan", args.dcgan_dir), ("patchgan", args.patchgan_dir)]:
        if gan_name not in args.conditions:
            continue
        if syn_dir is None or not syn_dir.exists():
            print(f"\nSkipping {gan_name.upper()}: --{gan_name}-dir not provided or missing.")
            continue

        for ratio in args.syn_ratios:
            n_syn = int(n_train_schiz * ratio)
            print(f"\n{gan_name.upper()} ratio={ratio:.0%} → adding {n_syn} synthetic images")

            try:
                syn_ds = SyntheticDataset(syn_dir, transform=base_tf, n_samples=n_syn)
            except RuntimeError as e:
                print(f"  Skipping: {e}")
                continue

            if len(syn_ds) < n_syn:
                print(
                    f"  Warning: only {len(syn_ds)} synthetic images available, "
                    f"requested {n_syn}."
                )

            train_real_ds = subset_with_transform(full_ds, train_idx, base_tf)
            mixed_train   = ConcatDataset([train_real_ds, syn_ds])

            cond_name = f"{gan_name.upper()} {int(ratio * 100)}%"
            metrics   = run_condition(
                cond_name,
                train_ds = mixed_train,
                val_ds   = subset_with_transform(full_ds, val_idx, base_tf),
                test_ds  = test_ds,
                args     = args,
                device   = device,
                out_dir  = args.out_dir,
            )
            all_results[cond_name]              = metrics
            proportion_results[gan_name][ratio] = metrics

    # Aggregate Plots
    # Summary figures are saved to both results/ and blog_figures/
    blog_dir = Path("blog_figures")
    blog_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Generating aggregate plots ──")
    if len(all_results) > 1:
        plot_roc_curves(all_results, args.out_dir / "roc_curves.png")
        plot_roc_curves(all_results, blog_dir / "roc_curves.png")

    plot_accuracy_bar(all_results, args.out_dir / "accuracy_bar.png")
    plot_accuracy_bar(all_results, blog_dir / "accuracy_bar.png")

    has_proportion_data = any(proportion_results[g] for g in proportion_results)
    if has_proportion_data:
        plot_proportion_lines(proportion_results, args.out_dir / "proportion_lines.png")
        plot_proportion_lines(proportion_results, blog_dir / "proportion_lines.png")

    # Results Table
    rows = [
        {
            "Condition":    cond,
            "Accuracy":     round(m["accuracy"],     4),
            "Balanced Acc": round(m["balanced_acc"], 4),
            "Sensitivity":  round(m["sensitivity"],  4),
            "Specificity":  round(m["specificity"],  4),
            "AUC":          round(m["auc"],           4),
        }
        for cond, m in all_results.items()
    ]
    df = pd.DataFrame(rows)

    csv_path = args.out_dir / "results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults table → {csv_path}")

    md_path = args.out_dir / "results_table.md"
    md_path.write_text(df.to_markdown(index=False))
    print(f"Markdown table → {md_path}")

    print("\n" + "=" * 62)
    print(df.to_string(index=False))
    print("=" * 62)


if __name__ == "__main__":
    main()
