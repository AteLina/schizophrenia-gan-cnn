# CNNs and GANs to Identify Schizophrenia

**Authors:** Melina Sarion, Alexander Hadesman, Nicholas Enbar-Salo

---

## Overview

This project investigates whether GAN-generated synthetic brain images can improve the performance of a CNN classifier for schizophrenia detection. We train a ResNet-18 classifier under eight experimental conditions ranging from no augmentation to GAN-augmented training sets at 25%, 50%, and 100% mixing ratios and compare two GAN architectures (DCGAN and PatchGAN) against traditional data augmentation. The pipeline covers MRI preprocessing, GAN training and image generation, CNN training with early stopping, and evaluation with ROC curves, AUC, sensitivity, and specificity.

---

## Introduction

Schizophrenia is a severe psychiatric disorder affecting approximately 1% of the global population. Diagnosis currently relies on clinical interviews and behavioral observation, which are inherently subjective and prone to inconsistency. Neuroimaging research, particularly structural MRI, has identified measurable brain abnormalities associated with schizophrenia, such as reduced gray matter volume, enlarged ventricles, and altered prefrontal connectivity. This raises the possibility of objective, imaging-based diagnostic support tools powered by machine learning.

A major obstacle in building such tools is the scarcity of labeled neuroimaging data. Clinical datasets are small due to privacy constraints, the cost of MRI acquisition, and the difficulty of recruiting patients. Small datasets cause CNNs to overfit and generalize poorly. We address this by using GANs to generate synthetic schizophrenia brain images and evaluating whether GAN-based augmentation improves classifier performance.

---

## Repository Structure

```
schizophrenia-gan-cnn/
├── src/
│   ├── preprocess.py        # MRI preprocessing pipeline
│   ├── make_slices.py       # Slice extraction (resume utility)
│   ├── augment.py           # GAN training and image generation
│   └── cnn_run.py           # CNN training and evaluation
├── notebooks/
│   ├── preprocess_main_cobre_ucla.ipynb   # Preprocessing notebook (Colab)
│   └── slice_main_cobre_ucla.ipynb        # Slicing notebook (Colab)
├── blog_figures/            # Final figures used in the blog post
├── results/                 # Training outputs (generated, not tracked)
└── data/                    # Data directory (not tracked, see below)
```

---

## Data

We use two publicly available neuroimaging datasets:

- **UCLA Consortium for Neuropsychiatric Phenomics (CNP)** — [https://openneuro.org/datasets/ds000030](https://openneuro.org/datasets/ds000030)
- **COBRE (Center for Biomedical Research Excellence)** — [http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html](http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html)

Both datasets require registration and agreement to a data use agreement before downloading. Raw data is not included in this repository.

After downloading, place data in the following structure:

```
data/
  slices/
    schizophrenia/   ← preprocessed PNG slices, schizophrenia subjects
    healthy/         ← preprocessed PNG slices, healthy controls
  augmented/
    dcgan/
      schizophrenia/ ← DCGAN synthetic images
    patchgan/
      schizophrenia/ ← PatchGAN synthetic images
```

---

## How to Reproduce Results

### Step 1 — Install dependencies

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn tabulate
```

### Step 2 — Preprocess the raw MRI data

Run the preprocessing notebook in Google Colab:

```
notebooks/preprocess_main_cobre_ucla.ipynb
```

This performs skull stripping, registration to MNI152 space, bias field correction, ComBat harmonization across scanner sites, z-score normalization, and extracts the middle 70 axial slices per subject as 256×256 PNG images. The underlying functions are in `src/preprocess.py`.

### Step 3 — Generate synthetic image pools

```bash
python src/augment.py --mode all
```

This trains a DCGAN and PatchGAN on the preprocessed schizophrenia slices and saves fixed pools of synthetic images to `data/augmented/`. The pools are frozen after generation to ensure all subsequent CNN experiments use identical synthetic data.

### Step 4 — Train and evaluate the CNN

```bash
python src/cnn_run.py \
    --real-schiz-dir data/slices/schizophrenia \
    --real-ctrl-dir  data/slices/healthy \
    --dcgan-dir      data/augmented/dcgan/schizophrenia \
    --patchgan-dir   data/augmented/patchgan/schizophrenia \
    --out-dir        results
```

This runs all eight experimental conditions automatically:

| Condition | Description |
|---|---|
| No Augmentation | ResNet-18 trained on real images only |
| Traditional | Real images + random flip, rotation, crop, color jitter |
| DCGAN 25% | Real images + 25% synthetic schiz images from DCGAN |
| DCGAN 50% | Real images + 50% synthetic schiz images from DCGAN |
| DCGAN 100% | Real images + 100% synthetic schiz images from DCGAN |
| PatchGAN 25% | Real images + 25% synthetic schiz images from PatchGAN |
| PatchGAN 50% | Real images + 50% synthetic schiz images from PatchGAN |
| PatchGAN 100% | Real images + 100% synthetic schiz images from PatchGAN |

Results are saved to `results/` and summary figures are saved to `blog_figures/`.

---

## Blog Figures

All figures in `blog_figures/` are generated by `src/cnn_run.py` (Step 4 above).

| Figure | Description | Generating code |
|---|---|---|
| `roc_curves.png` | ROC curves for all 8 conditions overlaid on one plot | `plot_roc_curves()` in `src/cnn_run.py` |
| `accuracy_bar.png` | Bar chart comparing accuracy and AUC across all conditions | `plot_accuracy_bar()` in `src/cnn_run.py` |
| `proportion_lines.png` | Line graph showing performance vs. synthetic data proportion (25/50/100%) for DCGAN and PatchGAN | `plot_proportion_lines()` in `src/cnn_run.py` |
| `sample_augments_DCGAN.png` | Example synthetic brain images generated by DCGAN | `src/augment.py` |
