# need to train and generate
# 1) Traditional Augmentation
# 2) DCGAN
# 3) StyleGAN

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from torchvision.utils import save_image
from PIL import Image

import subprocess
import argparse


# Paths
BASE_DIR       = Path("/content/drive/MyDrive/schizophrenia_gan")
real_data_path = BASE_DIR / "data/slices/schizophrenia"

AUG_DIR  = BASE_DIR / "data/augmented"
CKPT_DIR = BASE_DIR / "checkpoints"


# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# Dataset
class MRISliceDataset(Dataset):

    def __init__(self, data_dir: Path):

        self.paths = sorted(list(data_dir.glob("*.png")))

        if len(self.paths) == 0:
            raise RuntimeError(f"No PNG files found in {data_dir}")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),

            # DCGAN architecture below is built for 64x64
            transforms.Resize((64, 64)),

            transforms.ToTensor(),

            # Normalize to [-1,1]
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img = Image.open(self.paths[idx]).convert("L")

        return self.transform(img)


# DCGAN Discriminator
class DCGAN_Discriminator(nn.Module):

    def __init__(self, channels=1):

        super().__init__()

        self.conv1 = nn.Conv2d(channels, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(512)

        # 4x4 -> 1x1
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0)

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)

        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)

        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)

        x = self.conv5(x)

        return x


# DCGAN Generator
class DCGAN_Generator(nn.Module):

    def __init__(self, noise_dim=128, channels=1):

        super().__init__()

        self.conv1 = nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, channels, 4, 2, 1)

    def forward(self, x):

        x = x.view(x.size(0), -1, 1, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.tanh(self.conv5(x))

        return x


# DCGAN Training
def dcgan_train(num_epochs=50):

    (CKPT_DIR / "dcgan").mkdir(parents=True, exist_ok=True)

    batch_size = 16
    noise_dim = 128

    loss_criteria = nn.BCEWithLogitsLoss()

    G = DCGAN_Generator().to(DEVICE)
    D = DCGAN_Discriminator().to(DEVICE)

    G.train()
    D.train()

    opt_g = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    opt_d = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    dataset = MRISliceDataset(real_data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True
    )

    for epoch in range(num_epochs):

        for batch in dataloader:

            real_images = batch.to(DEVICE)

            b = real_images.size(0)

            # Train Discriminator
            opt_d.zero_grad()

            # REAL
            d_predictions_real = D(real_images)

            labels_real = torch.ones_like(d_predictions_real).to(DEVICE)

            loss_d_real = loss_criteria(
                d_predictions_real,
                labels_real
            )

            # FAKE
            noise = torch.randn(b, noise_dim).to(DEVICE)

            fake_images = G(noise).detach()

            d_predictions_fake = D(fake_images)

            labels_fake = torch.zeros_like(d_predictions_fake).to(DEVICE)

            loss_d_fake = loss_criteria(
                d_predictions_fake,
                labels_fake
            )

            loss_d = loss_d_real + loss_d_fake

            loss_d.backward()

            opt_d.step()

            # Train Generator
            opt_g.zero_grad()

            noise = torch.randn(b, noise_dim).to(DEVICE)

            fake_images = G(noise)

            d_predictions = D(fake_images)

            labels_generator = torch.ones_like(d_predictions).to(DEVICE)

            loss_g = loss_criteria(
                d_predictions,
                labels_generator
            )

            loss_g.backward()

            opt_g.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"loss_G={loss_g.item():.4f} "
            f"loss_D={loss_d.item():.4f}"
        )

        # Save samples
        if (epoch + 1) % 20 == 0:

            sample_dir = CKPT_DIR / "dcgan/samples"

            sample_dir.mkdir(exist_ok=True)

            with torch.no_grad():

                sample_noise = torch.randn(16, noise_dim).to(DEVICE)

                fake_samples = G(sample_noise)

                save_image(
                    fake_samples,
                    sample_dir / f"epoch_{epoch+1:04d}.png",
                    nrow=4,
                    normalize=True
                )

    torch.save(
        G.state_dict(),
        CKPT_DIR / "dcgan/dcgan_generator_saved.pt"
    )

    return G


# DCGAN Generation
def dcgan_generate(G, num_images=1000):

    G.eval()

    G.to(DEVICE)

    out_dir = AUG_DIR / "dcgan"

    out_dir.mkdir(parents=True, exist_ok=True)

    noise = torch.randn(num_images, 128).to(DEVICE)

    with torch.no_grad():

        fake_images = G(noise)

        for i, image in enumerate(fake_images):

            save_image(
                image,
                out_dir / f"dcgan_image_{i:05d}.png",
                normalize=True
            )


# StyleGAN2-ADA Training
def stylegan_train(kimg=200):

    if DEVICE.type == "cpu":
        print("Skipping StyleGAN training because CUDA is unavailable.")
        return

    (CKPT_DIR / "stylegan").mkdir(parents=True, exist_ok=True)

    if not Path("stylegan2-ada-pytorch").exists():

        subprocess.run([
            "git",
            "clone",
            "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
        ])

    ffhq_pkl = CKPT_DIR / "stylegan/ffhq.pkl"

    if not ffhq_pkl.exists():

        subprocess.run([
            "curl",
            "-L",
            "-o",
            str(ffhq_pkl),
            "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
        ])

    subprocess.run([
        "python",
        "stylegan2-ada-pytorch/train.py",

        f"--outdir={CKPT_DIR / 'stylegan'}",

        f"--data={real_data_path}",

        f"--resume={ffhq_pkl}",

        f"--kimg={kimg}",

        "--augpipe=ada",

        "--mirror=1",

        "--gpus=1",
    ])


# StyleGAN Generation
def stylegan_generate(num_images=1000):

    out_dir = AUG_DIR / "stylegan"

    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots = sorted(
        (CKPT_DIR / "stylegan").glob("**/network-snapshot-*.pkl")
    )

    if not snapshots:

        print("No StyleGAN snapshots found.")
        return

    latest_ckpt = snapshots[-1]

    subprocess.run([
        "python",
        "stylegan2-ada-pytorch/generate.py",

        f"--outdir={out_dir}",

        f"--seeds=0-{num_images - 1}",

        f"--network={latest_ckpt}",
    ])


# Traditional Augmentation
def traditional_augment():

    out_dir = AUG_DIR / "traditional"

    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([

        transforms.Grayscale(num_output_channels=1),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(10),

        transforms.RandomResizedCrop(
            64,
            scale=(0.85, 1.0)
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),

        transforms.ToTensor(),
    ])

    png_files = sorted(real_data_path.glob("*.png"))

    for path in png_files:

        image = Image.open(path).convert("L")

        image_augmented = transform(image)

        save_image(
            image_augmented,
            out_dir / f"{path.stem}_aug.png"
        )


# Main
def main_augment(
    dcgan_epochs=50,
    stylegan_kimg=200
):

    (AUG_DIR / "dcgan").mkdir(parents=True, exist_ok=True)

    (AUG_DIR / "stylegan").mkdir(parents=True, exist_ok=True)

    (AUG_DIR / "traditional").mkdir(parents=True, exist_ok=True)

    # DCGAN
    dcgan_ckpt = CKPT_DIR / "dcgan/dcgan_generator_saved.pt"

    if dcgan_ckpt.exists():

        print("Loading existing DCGAN weights...")

        G = DCGAN_Generator()

        G.load_state_dict(
            torch.load(
                dcgan_ckpt,
                map_location=DEVICE
            )
        )

        G.to(DEVICE)

    else:

        print("Training DCGAN...")

        G = dcgan_train(num_epochs=dcgan_epochs)

    print("Generating DCGAN images...")

    dcgan_generate(G)

    # StyleGAN
    print("Training StyleGAN2-ADA...")

    stylegan_train(kimg=stylegan_kimg)

    print("Generating StyleGAN images...")

    stylegan_generate()

    # Traditional Augmentation
    print("Generating traditional augmentations...")

    traditional_augment()

    print("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "traditional",
            "dcgan",
            "stylegan"
        ],
        help="Which augmentation pipeline to run"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of DCGAN training epochs"
    )

    parser.add_argument(
        "--kimg",
        type=int,
        default=200,
        help="StyleGAN2-ADA kimg value"
    )

    args = parser.parse_args()

    # Traditional Augmentation Only
    if args.mode == "traditional":

        print("Running traditional augmentation only...")

        traditional_augment()

    # DCGAN Only
    elif args.mode == "dcgan":

        print("Running DCGAN only...")

        dcgan_ckpt = CKPT_DIR / "dcgan/dcgan_generator_saved.pt"

        if dcgan_ckpt.exists():

            print("Loading existing DCGAN weights...")

            G = DCGAN_Generator()

            G.load_state_dict(
                torch.load(
                    dcgan_ckpt,
                    map_location=DEVICE
                )
            )

            G.to(DEVICE)

        else:

            print("Training DCGAN...")

            G = dcgan_train(num_epochs=args.epochs)

        print("Generating DCGAN images...")

        dcgan_generate(G)

    # StyleGAN Only
    elif args.mode == "stylegan":

        print("Running StyleGAN2-ADA only...")

        stylegan_train(kimg=args.kimg)

        stylegan_generate()

    # Run Everything
    else:

        print("Running full augmentation pipeline...")

        main_augment(
            dcgan_epochs=args.epochs,
            stylegan_kimg=args.kimg
        )