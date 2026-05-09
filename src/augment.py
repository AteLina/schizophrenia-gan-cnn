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

# path object for original data
from pathlib import Path

from torchvision.utils import save_image
from PIL import Image

import subprocess

# ── Paths (Google Drive) ──────────────────────────────────────────────────────
BASE_DIR = Path("/content/drive/MyDrive/schizophrenia_gan")
real_data_path = BASE_DIR / "data/slices/schizophrenia"  # GAN trains on SCZ only
AUG_DIR = BASE_DIR / "data/augmented"
CKPT_DIR = BASE_DIR / "checkpoints"

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class MRISliceDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.paths = sorted(list(data_dir.glob("*.png")))
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # scale to [-1, 1] for GAN
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img)


# DCGAN Class
class DCGAN_Discriminator(nn.Module):
    def __init__(self, channels=1):  # channels = 1 for greyscale MRI images
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0)

    def forward(self, x):
        # define forward pass here
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.conv5(x)
        return x


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
        # need to reshape the noise vector so we can use in conv1
        x = x.view(x.size(0), -1, 1, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # we use tanh so we can normalize pixel values between -1 and 1
        x = torch.tanh(self.conv5(x))
        return x

    # ---------#


def dcgan_train():
    # making sure proper checkpoint directory exists
    (CKPT_DIR / "dcgan").mkdir(parents=True, exist_ok=True)

    # first, we define hyperparameters, initialize models, and define optimizers and loss criteria
    num_epochs = 200
    batch_size = 16
    noise_dim = 128

    loss_criteria = nn.BCEWithLogitsLoss()

    G = DCGAN_Generator().to(DEVICE)
    D = DCGAN_Discriminator().to(DEVICE)

    opt_g = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    dataset = MRISliceDataset(real_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, drop_last=True)

    for epoch in range(num_epochs):
        for batch in dataloader:
            real_images = batch.to(DEVICE)
            b = real_images.size(0)  # actual batch size

            # -----------------#
            # Train Discriminator
            opt_d.zero_grad()
            # Discriminator Loss

            # real -> make sure we can correctly label real images
            labels_real = torch.ones(b, 1, 1, 1).to(DEVICE)
            d_predictions = D(real_images)
            loss_d_real = loss_criteria(d_predictions, labels_real)

            # fake -> make sure we can correctly label fake images
            # feed Generator noise to have it create fake images, then make sure Discriminator accurately labels fake images as false
            noise = torch.randn(b, noise_dim).to(DEVICE)
            fake_images = G(noise).detach()  # detach so G gradients don't flow into D
            labels_fake = torch.zeros(b, 1, 1, 1).to(DEVICE)
            d_predictions_fake = D(fake_images)
            loss_d_fake = loss_criteria(d_predictions_fake, labels_fake)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # ------------------#
            # Train Generator -> goal is to trick Discriminator
            opt_g.zero_grad()
            noise = torch.randn(b, noise_dim).to(DEVICE)
            fake_images = G(noise)
            d_predictions = D(fake_images)
            # goal = have Discriminator label fake images as real (1)
            labels_generator = torch.ones(b, 1, 1, 1).to(DEVICE)

            loss_g = loss_criteria(d_predictions, labels_generator)
            loss_g.backward()
            opt_g.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] loss_G={loss_g.item():.4f} loss_D={loss_d.item():.4f}")

        # Save sample grid every 20 epochs to monitor for mode collapse
        if (epoch + 1) % 20 == 0:
            sample_dir = CKPT_DIR / "dcgan/samples"
            sample_dir.mkdir(exist_ok=True)
            with torch.no_grad():
                sample_noise = torch.randn(16, noise_dim).to(DEVICE)
                save_image(G(sample_noise), sample_dir / f"epoch_{epoch + 1:04d}.png",
                           nrow=4, normalize=True)

    # after training, we will save Generator weights so that we do not have to retrain every time.
    torch.save(G.state_dict(), CKPT_DIR / "dcgan/dcgan_generator_saved.pt")

    return G


# num_images tells us how many "fake" images we want the dcgan to generate, by default.
def dcgan_generate(G, num_images=1000):
    G.eval()  # dcgan_generate is just generating images, we don't want to update Generator weights
    G.to(DEVICE)

    out_dir = AUG_DIR / "dcgan"
    out_dir.mkdir(parents=True, exist_ok=True)

    noise = torch.randn(num_images, 128).to(DEVICE)

    with torch.no_grad():  # we can save memory by telling Pytorch not to store gradients during the image generation
        fake_images = G(noise)

        for i, image in enumerate(fake_images):
            save_image(image, out_dir / f"image_{i}.png", normalize=True)


def stylegan_train():
    # convention for using StyleGAN2-ADA. Subprocess allows us to run StyleGAN2 training from command line.
    # we are using starting weights from pretrained FFHQ model.

    # ensuring proper checkpoints directory exists
    (CKPT_DIR / "stylegan").mkdir(parents=True, exist_ok=True)

    # clone repo if not already present
    if not Path("stylegan2-ada-pytorch").exists():
        subprocess.run(["git", "clone",
                        "https://github.com/NVlabs/stylegan2-ada-pytorch.git"])

    # download FFHQ pretrained weights if not already present
    ffhq_pkl = CKPT_DIR / "stylegan/ffhq.pkl"
    if not ffhq_pkl.exists():
        subprocess.run([
            "curl", "-L", "-o", str(ffhq_pkl),
            "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
        ])

    subprocess.run([
        "python", "stylegan2-ada-pytorch/train.py",
        f"--outdir={CKPT_DIR / 'stylegan'}",
        f"--data={real_data_path}",
        f"--resume={ffhq_pkl}",
        "--kimg=1000",
        "--augpipe=ada",  # adaptive augmentation -- designed for small datasets like ours
        "--mirror=1",
        "--gpus=1",
    ])


def stylegan_generate(num_images=1000):
    out_dir = AUG_DIR / "stylegan"
    out_dir.mkdir(parents=True, exist_ok=True)

    # find latest snapshot
    snapshots = sorted((CKPT_DIR / "stylegan").glob("network-snapshot-*.pkl"))
    if not snapshots:
        raise FileNotFoundError("No StyleGAN2 snapshots found. Run stylegan_train() first.")
    latest_ckpt = snapshots[-1]

    subprocess.run([
        "python", "stylegan2-ada-pytorch/generate.py",
        f"--outdir={out_dir}",
        f"--seeds=0-{num_images - 1}",
        f"--network={latest_ckpt}",
    ])


def traditional_augment():
    # for traditional augmentation, we will use combination of 3 transformations: rotation, zoom, and brightness and contrast
    # we will exactly double the amount of data, as we are applying the transform once to each image in real_images.

    out_dir = AUG_DIR / "traditional"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(256, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    png_files = sorted(real_data_path.glob("*.png"))

    # here, we are essentially doubling the amount of data by applying transform once to every image in real_images
    for i, path in enumerate(png_files):
        image = Image.open(path).convert("L")
        image_augmented = transform(image)
        save_image(image_augmented, out_dir / f"image_{i}.png")


def main_augment():
    # main_augment() runs DCGAN and StyleGAN training and generation + traditional augmentation.

    # making sure proper directories exist
    (AUG_DIR / "dcgan").mkdir(parents=True, exist_ok=True)
    (AUG_DIR / "stylegan").mkdir(parents=True, exist_ok=True)
    (AUG_DIR / "traditional").mkdir(parents=True, exist_ok=True)

    # DCGAN Training + Generation

    # if we have already trained the Generator and have the weights saved, we just load the weights and skip retraining.
    # Otherwise, we need to train the dcGAN.
    if (CKPT_DIR / "dcgan/dcgan_generator_saved.pt").exists():
        G = DCGAN_Generator()
        G.load_state_dict(torch.load(CKPT_DIR / "dcgan/dcgan_generator_saved.pt",
                                     map_location=DEVICE))
    else:
        G = dcgan_train()

    dcgan_generate(G)

    # StyleGAN Training + Generation
    stylegan_train()
    stylegan_generate()

    # Traditional Augmentation
    traditional_augment()


if __name__ == "__main__":
    main_augment()