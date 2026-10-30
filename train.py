import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

#from DiffusionCNN import SinusoidalPositionEmbeddings, SimpleCNNBlock, SimpleDDPMCNN
#from DiffusionUNet import SinusoidalPositionEmbeddings, Block, DDPMResidualBDDPMUNetlock, SelfAttention, DDPMUNet
from DiffusionCNN import SimpleDDPMCNN
from DiffusionUNet import DDPMUNet

# -------------------- DDPM noise schedule --------------------

class DiffusionSchedule:
    def __init__(self, timesteps, beta_start, beta_end, device):
        self.timesteps = timesteps
        self.betas = torch.linspace(
            beta_start, beta_end, timesteps, device=device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(self, clean_images, timesteps, noise):
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (
            self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)
        )

        return (
            sqrt_alpha_bar * clean_images
            + sqrt_one_minus_alpha_bar * noise
        )

# -------------------- Image-folder dataset --------------------

class ImageFolder(Dataset):
    """Load image files recursively as 128x128."""

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, folder):
        self.paths = sorted(
            path
            for path in Path(folder).rglob("*")
            if path.is_file() and path.suffix.lower() in self.EXTENSIONS
        )

        if not self.paths:
            raise FileNotFoundError(f"No supported images found in: {folder}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image)

# -------------------- Training and evaluation --------------------

def get_dataset(device, dataset, model, directory, validation_size, batch_size, workers, seed):

    if dataset == "mnist":
        image_size = 32
        image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

        train_set = datasets.MNIST(root=directory, train=True, download=True, transform=image_transform,)
        test_set = datasets.MNIST(root=directory, train=False, download=True, transform=image_transform,)
    else:
        all_images = ImageFolder(directory)
        test_size = max(1, int(len(all_images) * 0.1))
        train_set, test_set = random_split(all_images, [len(all_images) - test_size, test_size], generator=torch.Generator().manual_seed(seed))

    validation_size = max(1, int(len(train_set) * validation_size))
    if validation_size >= len(train_set):
        parser.error("The training set needs at least two images.")

    train_size = len(train_set) - validation_size
    training_set, validation_set = random_split(
        train_set,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )

    loader_args = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(training_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    return train_loader, validation_loader, test_loader

def run_epoch(model, loader, schedule, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    image_count = 0

    for batch_data in loader:
        if isinstance(batch_data, (tuple, list)): batch_data = batch_data[0] 
        clean_images = batch_data.to(
            device, non_blocking=True
        )
        batch_size = clean_images.shape[0]

        # Sample one timestep and one total noise image for each training image.
        timesteps = torch.randint(
            0,
            schedule.timesteps,
            (batch_size,),
            device=device,
        )
        noise = torch.randn_like(clean_images)
        noisy_images = schedule.add_noise(
            clean_images, timesteps, noise
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        predicted_noise = model(noisy_images, timesteps)
        loss = F.mse_loss(predicted_noise, noise)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * batch_size
        image_count += batch_size

    return total_loss / max(image_count, 1)

def train_model(device, model, train_loader, validation_loader, test_set_size, T, beta_start, beta_end, width, schedule, epochs, optimizer, checkpoint_path):
    best_validation_loss = float("inf")
    train_history = []
    validation_history = []

    print(f"Device: {device}")
    print(
        f"Images: train={len(train_loader.dataset)}, "
        f"validation={len(validation_loader.dataset)}, test={test_set_size}"
    )
    print(f"Diffusion steps T: {T}")
    print(f"Linear beta schedule: {beta_start} to {beta_end}")
    print(f"Final cumulative alpha_bar: {schedule.alpha_bars[-1].item():.8f}")

    for epoch_number in range(1, epochs + 1):
        train_loss = run_epoch(
            model, train_loader, schedule, device, optimizer
        )

        # Reuse the same validation noise and timesteps at every epoch.
        torch.manual_seed(9876)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(9876)

        validation_loss = run_epoch(model, validation_loader, schedule, device)

        train_history.append(train_loss)
        validation_history.append(validation_loss)

        print(
            f"Epoch {epoch_number:03d}/{epochs} | "
            f"train noise MSE={train_loss:.6f} | "
            f"validation noise MSE={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "width": width,
                    "T": T,
                    "beta_start": beta_start,
                    "beta_end": beta_end,
                    "best_validation_noise_mse": best_validation_loss,
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint: {checkpoint_path}")

    return train_history, validation_history

def test_model(device, model, test_loader, schedule, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    torch.manual_seed(54321)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(54321)

    test_loss = run_epoch(model, test_loader, schedule, device)
    print(f"Test noise-prediction MSE: {test_loss:.6f}")
    return test_loss

# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Train a standard DDPM noise-prediction CNN or U-Net.")

    parser.add_argument("--dataset", choices=["mnist", "folders"], default="mnist", help="Use torchvision MNIST or separate image folders.")
    parser.add_argument("--model", choices=["cnn", "unet"], default="cnn")
    parser.add_argument("--data-dir", default="./data")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=64)

    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)

    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.T < 2:
        parser.error("--T must be at least 2.")
    if not (0.0 < args.beta_start < args.beta_end < 1.0):
        parser.error("Require 0 < beta-start < beta-end < 1.")
    if args.width % 8 != 0:
        parser.error("--width must be divisible by 8.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():  torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader, test_loader = get_dataset(
        device,
        args.dataset, 
        args.model, 
        args.data_dir, 
        args.val_fraction, 
        args.batch_size, 
        args.workers, 
        args.seed)

    schedule = DiffusionSchedule(
        timesteps=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    channel = 3
    if args.dataset == "mnist":
        channel = 1

    model = SimpleDDPMCNN(in_channels=channel, out_channels=channel, hidden_dim=args.width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    if args.model == "unet":
        model = DDPMUNet(in_channels=channel, out_channels=channel, base_channels=args.width).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
            )

    checkpoint_path = Path(f"models/ddpm_{args.model}_{args.dataset if args.dataset == "mnist" else "image"}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_history, validation_history = train_model(
        device, 
        model, 
        train_loader, 
        validation_loader, 
        len(test_loader.dataset), 
        args.T, 
        args.beta_start, 
        args.beta_end, 
        args.width, 
        schedule, 
        args.epochs, 
        optimizer, 
        checkpoint_path
        )

    test_loss = test_model(
        device, 
        model, 
        test_loader, 
        schedule, 
        checkpoint_path
        )

    plt.figure(figsize=(9, 5))
    epochs = range(1, args.epochs + 1)
    plt.plot(epochs, train_history, label="Training noise MSE")
    plt.plot(epochs, validation_history, label="Validation noise MSE")
    plt.axhline(
        test_loss,
        color="green",
        linestyle="--",
        label=f"Test noise MSE ({test_loss:.4f})",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Total-noise prediction MSE")
    plt.title(f"DDPM {args.model.upper()} training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    graph_path = Path(f"results/{checkpoint_path.with_name(
        checkpoint_path.stem + "_mse.png"
    )}")
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(f"{graph_path}", dpi=150)
    print(f"Saved MSE graph: {graph_path}")
    plt.show()

if __name__ == "__main__":
    main()