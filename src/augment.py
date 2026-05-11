"""
GAN training and synthetic image generation for MRI augmentation.

Augmentation methods (run all three or choose one with --mode):
    1. Traditional augmentation   — random flip, rotation, crop, color jitter
    2. DCGAN                      — conditional DCGAN, 500 images per class
    3. PatchGAN                   — conditional PatchGAN, 500 images per class

Usage:
    python augment.py --mode all
    python augment.py --mode dcgan    --epochs 50 --num_images 500
    python augment.py --mode patchgan --epochs 50 --num_images 500
    python augment.py --mode traditional
"""

# Imports
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from torchvision.utils import save_image
from PIL import Image

import argparse

# Paths
BASE_DIR = Path("/content/drive/MyDrive/schizophrenia_gan")
HEALTHY_DIR = BASE_DIR / "data/slices/healthy"
SCHIZO_DIR = BASE_DIR / "data/slices/schizophrenia"

AUG_DIR = BASE_DIR / "data/augmented"
CKPT_DIR = BASE_DIR / "checkpoints"

# Create labels and other constants
LABEL_HEALTHY = 0
LABEL_SCHIZO = 1
NUM_CLASSES = 2
CLASS_NAMES = {LABEL_HEALTHY: "healthy", LABEL_SCHIZO: "schizophrenia"}

# Backend work, ensure cuda is used if available to minimize runtime
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# Loads PNG brain MRI slices from healthy/ and schizophrenia/ directories
class MRISliceDataset(Dataset):

    def __init__(self, healthy_dir: Path, schizo_dir: Path):

        healthy_paths = sorted(healthy_dir.glob("*.png"))
        schizo_paths = sorted(schizo_dir.glob("*.png"))

        if not healthy_paths:
            raise RuntimeError(f"No PNG files found in {healthy_dir}")
        if not schizo_paths:
            raise RuntimeError(f"No PNG files found in {schizo_dir}")

        # (path, label) pairs
        self.samples = (
                [(p, LABEL_HEALTHY) for p in healthy_paths] +
                [(p, LABEL_SCHIZO) for p in schizo_paths]
        )

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # → [-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), label

# DCGAN

# DCGAN discriminator judges whether an image is real or fake, conditioned on class label
class DCGAN_Discriminator(nn.Module):

    def __init__(self, channels: int = 1, num_classes: int = NUM_CLASSES, img_size: int = 64):
        super().__init__()

        self.img_size = img_size

        # One embedding vector per class, projected to a full feature map
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        # Input: image channel + label channel = channels + 1
        self.conv1 = nn.Conv2d(channels + 1, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0) # 4×4 → 1×1

    def forward(self, x, labels):
        b = x.size(0)

        # Embed label → (b, 1, H, W)
        label_map = self.label_emb(labels).view(b, 1, self.img_size, self.img_size)

        x = torch.cat([x, label_map], dim=1)

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.conv5(x)

        return x

# DCGAN generator creates synthetic brain images from noise and class label
class DCGAN_Generator(nn.Module):

    def __init__(self, noise_dim: int = 128, channels: int = 1, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.noise_dim = noise_dim
        self.label_dim = 32 # size of the label embedding
        in_dim = noise_dim + self.label_dim

        self.label_emb = nn.Embedding(num_classes, self.label_dim)

        self.conv1 = nn.ConvTranspose2d(in_dim, 512, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, channels, 4, 2, 1)

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels)
        x = torch.cat([noise, label_emb], dim=1)
        x = x.view(x.size(0), -1, 1, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.tanh(self.conv5(x))

        return x

# Trains the DCGAN on the full MRI dataset, saves weights, returns the generator
def dcgan_train(num_epochs: int = 50) -> DCGAN_Generator:
    (CKPT_DIR / "dcgan").mkdir(parents=True, exist_ok=True)

    batch_size = 16
    noise_dim = 128

    loss_fn = nn.BCEWithLogitsLoss()

    G = DCGAN_Generator(noise_dim=noise_dim).to(DEVICE)
    D = DCGAN_Discriminator().to(DEVICE)

    G.train()
    D.train()

    opt_g = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    dataset = MRISliceDataset(HEALTHY_DIR, SCHIZO_DIR)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True, pin_memory=True
    )

    for epoch in range(num_epochs):

        for real_images, labels in dataloader:
            real_images = real_images.to(DEVICE)
            labels = labels.to(DEVICE)
            b = real_images.size(0)

            # Discriminator
            opt_d.zero_grad()

            pred_real = D(real_images, labels)
            loss_d_real = loss_fn(pred_real, torch.ones_like(pred_real))

            noise = torch.randn(b, noise_dim, device=DEVICE)
            fake_imgs = G(noise, labels).detach()
            pred_fake = D(fake_imgs, labels)
            loss_d_fake = loss_fn(pred_fake, torch.zeros_like(pred_fake))

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # Generator
            opt_g.zero_grad()

            noise = torch.randn(b, noise_dim, device=DEVICE)
            fake_imgs = G(noise, labels)
            pred = D(fake_imgs, labels)
            loss_g = loss_fn(pred, torch.ones_like(pred))

            loss_g.backward()
            opt_g.step()

        print(
            f"[DCGAN] Epoch [{epoch + 1}/{num_epochs}]  "
            f"loss_G={loss_g.item():.4f}  loss_D={loss_d.item():.4f}"
        )

        # Save sample grid every 20 epochs
        if (epoch + 1) % 20 == 0:
            sample_dir = CKPT_DIR / "dcgan/samples"
            sample_dir.mkdir(exist_ok=True)

            with torch.no_grad():
                # 8 healthy + 8 schizophrenia
                sample_noise = torch.randn(16, noise_dim, device=DEVICE)
                sample_labels = torch.tensor(
                    [LABEL_HEALTHY] * 8 + [LABEL_SCHIZO] * 8, device=DEVICE
                )
                fake_samples = G(sample_noise, sample_labels)
                save_image(
                    fake_samples,
                    sample_dir / f"epoch_{epoch + 1:04d}.png",
                    nrow=8, normalize=True
                )

    torch.save(G.state_dict(), CKPT_DIR / "dcgan/dcgan_generator_saved.pt")

    return G

# Generates num_images_per_class synthetic images per class, saves to data/augmented/dcgan/
def dcgan_generate(G: DCGAN_Generator, num_images_per_class: int = 500):

    G.eval()
    G.to(DEVICE)

    noise_dim = G.noise_dim

    for label_idx, class_name in CLASS_NAMES.items():

        out_dir = AUG_DIR / "dcgan" / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = torch.full((num_images_per_class,), label_idx,
                            dtype=torch.long, device=DEVICE)
        noise = torch.randn(num_images_per_class, noise_dim, device=DEVICE)

        with torch.no_grad():
            fake_images = G(noise, labels)

        for i, img in enumerate(fake_images):
            save_image(img, out_dir / f"dcgan_{class_name}_{i:05d}.png", normalize=True)

        print(f"[DCGAN] Saved {num_images_per_class} images → {out_dir}")

# PatchGAN

# PatchGAN discriminator scores overlapping 70×70 patches as real or fake, conditioned on class label
class PatchGAN_Discriminator(nn.Module):

    def __init__(self, channels: int = 1, num_classes: int = NUM_CLASSES, img_size: int = 64):
        super().__init__()

        self.img_size = img_size

        # Label → spatial feature map (same as DCGAN_D)
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        def block(in_c, out_c, stride, normalise=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False)]
            if normalise:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Input channels = image channel + label channel
        self.model = nn.Sequential(
            block(channels + 1, 64, stride=2, normalise=False),  # 64 → 32
            block(64, 128, stride=2),  # 32 → 16
            block(128, 256, stride=2),  # 16 →  8
            block(256, 512, stride=1),  # 8 →  8
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # 8 →  7
        )

    def forward(self, x, labels):
        b = x.size(0)
        label_map = self.label_emb(labels).view(b, 1, self.img_size, self.img_size)
        x = torch.cat([x, label_map], dim=1)
        return self.model(x)


# PatchGAN generator creates synthetic brain images using upsampling blocks
class PatchGAN_Generator(nn.Module):

    def __init__(self, noise_dim: int = 128, channels: int = 1, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.noise_dim = noise_dim
        self.label_dim = 32
        in_dim = noise_dim + self.label_dim

        self.label_emb = nn.Embedding(num_classes, self.label_dim)

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_c, affine=True),
                nn.ReLU(inplace=True),
            )

        self.proj = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 512, 4, 1, 0, bias=False),  # → 4×4
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
        )

        self.up1 = up_block(512, 256)  # 4  →  8
        self.up2 = up_block(256, 128)  # 8  → 16
        self.up3 = up_block(128, 64)  # 16 → 32
        self.up4 = up_block(64, 32)  # 32 → 64

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels)
        x = torch.cat([noise, label_emb], dim=1).view(-1, self.noise_dim + self.label_dim, 1, 1)

        x = self.proj(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        return self.out_conv(x)

# Trains the PatchGAN on the full MRI dataset, saves weights, returns the generator
def patchgan_train(num_epochs: int = 50) -> PatchGAN_Generator:
    (CKPT_DIR / "patchgan").mkdir(parents=True, exist_ok=True)

    batch_size = 16
    noise_dim = 128

    # LSGAN loss (MSE) works better with PatchGAN than BCE
    loss_fn = nn.MSELoss()

    G = PatchGAN_Generator(noise_dim=noise_dim).to(DEVICE)
    D = PatchGAN_Discriminator().to(DEVICE)

    G.train()
    D.train()

    opt_g = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    dataset = MRISliceDataset(HEALTHY_DIR, SCHIZO_DIR)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True, pin_memory=True
    )

    for epoch in range(num_epochs):

        for real_images, labels in dataloader:
            real_images = real_images.to(DEVICE)
            labels = labels.to(DEVICE)
            b = real_images.size(0)

            # Discriminator
            opt_d.zero_grad()

            pred_real = D(real_images, labels)
            # loss, real = 1, fake = 0
            loss_d_real = loss_fn(pred_real, torch.ones_like(pred_real))

            noise = torch.randn(b, noise_dim, device=DEVICE)
            fake_imgs = G(noise, labels).detach()
            pred_fake = D(fake_imgs, labels)
            loss_d_fake = loss_fn(pred_fake, torch.zeros_like(pred_fake))

            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # Generator
            opt_g.zero_grad()

            noise = torch.randn(b, noise_dim, device=DEVICE)
            fake_imgs = G(noise, labels)
            pred = D(fake_imgs, labels)
            loss_g = loss_fn(pred, torch.ones_like(pred))

            loss_g.backward()
            opt_g.step()

        print(
            f"[PatchGAN] Epoch [{epoch + 1}/{num_epochs}]  "
            f"loss_G={loss_g.item():.4f}  loss_D={loss_d.item():.4f}"
        )

        # Save sample grid every 20 epochs
        if (epoch + 1) % 20 == 0:
            sample_dir = CKPT_DIR / "patchgan/samples"
            sample_dir.mkdir(exist_ok=True)

            with torch.no_grad():
                sample_noise = torch.randn(16, noise_dim, device=DEVICE)
                sample_labels = torch.tensor(
                    [LABEL_HEALTHY] * 8 + [LABEL_SCHIZO] * 8, device=DEVICE
                )
                fake_samples = G(sample_noise, sample_labels)
                save_image(
                    fake_samples,
                    sample_dir / f"epoch_{epoch + 1:04d}.png",
                    nrow=8, normalize=True
                )

    torch.save(G.state_dict(), CKPT_DIR / "patchgan/patchgan_generator_saved.pt")

    return G

# Generates num_images_per_class synthetic images per class, saves to data/augmented/patchgan/
def patchgan_generate(G: PatchGAN_Generator, num_images_per_class: int = 500):

    G.eval()
    G.to(DEVICE)

    noise_dim = G.noise_dim

    for label_idx, class_name in CLASS_NAMES.items():

        out_dir = AUG_DIR / "patchgan" / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = torch.full((num_images_per_class,), label_idx,
                            dtype=torch.long, device=DEVICE)
        noise = torch.randn(num_images_per_class, noise_dim, device=DEVICE)

        with torch.no_grad():
            fake_images = G(noise, labels)

        for i, img in enumerate(fake_images):
            save_image(img, out_dir / f"patchgan_{class_name}_{i:05d}.png", normalize=True)

        print(f"[PatchGAN] Saved {num_images_per_class} images → {out_dir}")

# Traditional Augmentation

# Applies random transforms to each real image and saves augmented copies to data/augmented/traditional/
def traditional_augment():

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(64, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    sources = {
        LABEL_HEALTHY: HEALTHY_DIR,
        LABEL_SCHIZO: SCHIZO_DIR,
    }

    for label_idx, src_dir in sources.items():
        class_name = CLASS_NAMES[label_idx]
        out_dir = AUG_DIR / "traditional" / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        png_files = sorted(src_dir.glob("*.png"))

        for path in png_files:
            img = Image.open(path).convert("L")
            aug_img = transform(img)
            save_image(aug_img, out_dir / f"{path.stem}_aug.png")

        print(f"[Traditional] Augmented {len(png_files)} images → {out_dir}")

# Full Pipeline

# Loads existing DCGAN weights if available, otherwise trains from scratch
def _load_or_train_dcgan(epochs: int) -> DCGAN_Generator:
    ckpt = CKPT_DIR / "dcgan/dcgan_generator_saved.pt"
    G = DCGAN_Generator()

    if ckpt.exists():
        print("[DCGAN] Loading existing weights...")
        G.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        G.to(DEVICE)
    else:
        print("[DCGAN] Training from scratch...")
        G = dcgan_train(num_epochs=epochs)

    return G

# Loads existing PatchGAN weights if available, otherwise trains from scratch
def _load_or_train_patchgan(epochs: int) -> PatchGAN_Generator:
    ckpt = CKPT_DIR / "patchgan/patchgan_generator_saved.pt"
    G = PatchGAN_Generator()

    if ckpt.exists():
        print("[PatchGAN] Loading existing weights...")
        G.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        G.to(DEVICE)
    else:
        print("[PatchGAN] Training from scratch...")
        G = patchgan_train(num_epochs=epochs)

    return G

# Runs the full augmentation pipeline: DCGAN, PatchGAN, and traditional
def main_augment(dcgan_epochs: int = 50, patchgan_epochs: int = 50):
    # DCGAN
    G_dcgan = _load_or_train_dcgan(dcgan_epochs)
    print("[DCGAN] Generating images...")
    dcgan_generate(G_dcgan)

    # PatchGAN
    G_patch = _load_or_train_patchgan(patchgan_epochs)
    print("[PatchGAN] Generating images...")
    patchgan_generate(G_patch)

    # Traditional
    print("[Traditional] Augmenting images...")
    traditional_augment()

    print("\nAll done. Output tree:")
    print(f"  {AUG_DIR}/")
    print(f"  ├── dcgan/healthy/          ← {500} images")
    print(f"  ├── dcgan/schizophrenia/    ← {500} images")
    print(f"  ├── patchgan/healthy/       ← {500} images")
    print(f"  ├── patchgan/schizophrenia/ ← {500} images")
    print(f"  ├── traditional/healthy/    ← one augmented copy per real image")
    print(f"  └── traditional/schizophrenia/")

# Parse command-line arguments

# Main

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MRI slice augmentation pipeline")

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "traditional", "dcgan", "patchgan"],
        help="Which augmentation pipeline to run (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for DCGAN and/or PatchGAN (default: 50)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=500,
        help="Number of synthetic images to generate *per class* for GAN methods (default: 500)",
    )

    args = parser.parse_args()

    if args.mode == "traditional":
        print("Running traditional augmentation only...")
        traditional_augment()

    elif args.mode == "dcgan":
        print("Running DCGAN only...")
        G = _load_or_train_dcgan(args.epochs)
        dcgan_generate(G, num_images_per_class=args.num_images)

    elif args.mode == "patchgan":
        print("Running PatchGAN only...")
        G = _load_or_train_patchgan(args.epochs)
        patchgan_generate(G, num_images_per_class=args.num_images)

    else:
        print("Running full augmentation pipeline...")
        main_augment(
            dcgan_epochs=args.epochs,
            patchgan_epochs=args.epochs,
        )
