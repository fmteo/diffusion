import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import os
import sys

# Add the parent directory (project root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DiffusionCNN import SimpleDDPMCNN
from DiffusionUNet import DDPMUNet
from train import DiffusionSchedule, test_model as evaluate_checkpoint

TIMESTEPS = 8

def make_schedule():
    return DiffusionSchedule(
        timesteps=TIMESTEPS,
        beta_start=1e-4,
        beta_end=0.02,
        device=torch.device("cpu"),
    )

def make_model(model_name):
    if model_name == "cnn":
        # The CNN preserves the spatial dimensions; test it on MNIST-sized input.
        return SimpleDDPMCNN(
            in_channels=1,
            out_channels=1,
            hidden_dim=8,
        ), (2, 1, 28, 28)

    if model_name == "unet":
        # The training script resizes MNIST to 32x32 for its U-Net.
        return DDPMUNet(
            in_channels=1,
            out_channels=1,
            base_channels=8,
        ), (2, 1, 32, 32)

    raise ValueError(f"Unknown model: {model_name}")

def test_diffusion_schedule_is_valid():
    schedule = make_schedule()

    assert torch.isfinite(schedule.betas).all()
    assert torch.isfinite(schedule.alpha_bars).all()
    assert torch.all(schedule.betas > 0)
    assert torch.all(schedule.betas < 1)
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])

def test_forward_noise_matches_ddpm_formula():
    schedule = make_schedule()
    clean = torch.rand(2, 1, 32, 32) * 2 - 1
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 5], dtype=torch.long)

    actual = schedule.add_noise(clean, timesteps, noise)

    alpha_bar = schedule.alpha_bars[timesteps].view(-1, 1, 1, 1)
    expected = (
        torch.sqrt(alpha_bar) * clean
        + torch.sqrt(1.0 - alpha_bar) * noise
    )

    assert actual.shape == clean.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)

def reverse_diffusion_smoke_test(model, noisy_image, schedule):
    """Run standard DDPM reverse steps and return a display-range image."""
    model.eval()
    image = noisy_image.clone()

    with torch.no_grad():
        for t in reversed(range(schedule.timesteps)):
            timesteps = torch.full(
                (image.shape[0],),
                t,
                device=image.device,
                dtype=torch.long,
            )

            predicted_noise = model(image, timesteps)

            beta_t = schedule.betas[t]
            alpha_t = schedule.alphas[t]
            alpha_bar_t = schedule.alpha_bars[t]

            mean = (
                image
                - beta_t / torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_t)

            if t > 0:
                alpha_bar_previous = schedule.alpha_bars[t - 1]
                variance = (
                    beta_t * (1.0 - alpha_bar_previous)
                    / (1.0 - alpha_bar_t)
                ).clamp_min(0.0)

                image = mean + torch.sqrt(variance) * torch.randn_like(image)
            else:
                image = mean

    return image.clamp(-1.0, 1.0)

@pytest.mark.parametrize("model_name", ["cnn", "unet"])
def test_model_output_matches_input_shape(model_name):
    model, input_shape = make_model(model_name)
    model.eval()

    noisy_image = torch.randn(*input_shape)
    timesteps = torch.tensor([1, 6], dtype=torch.long)

    with torch.no_grad():
        predicted_noise = model(noisy_image, timesteps)

    assert predicted_noise.shape == noisy_image.shape
    assert torch.isfinite(predicted_noise).all()

@pytest.mark.parametrize("model_name", ["cnn", "unet"])
def test_model_can_complete_training_step(model_name):
    schedule = make_schedule()
    model, input_shape = make_model(model_name)
    model.train()

    clean = torch.rand(*input_shape) * 2 - 1
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([1, 6], dtype=torch.long)
    noisy = schedule.add_noise(clean, timesteps, noise)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    predicted_noise = model(noisy, timesteps)
    loss = F.mse_loss(predicted_noise, noise)
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    optimizer.step()

@pytest.mark.parametrize("model_name", ["cnn", "unet"])
def test_checkpoint_loads_and_evaluates(model_name, tmp_path):
    device = torch.device("cpu")
    schedule = make_schedule()
    model, input_shape = make_model(model_name)
    model.eval()

    checkpoint_path = tmp_path / f"{model_name}_checkpoint.pt"
    torch.save(
        {"model_state_dict": model.state_dict()},
        checkpoint_path,
    )

    clean = torch.rand(*input_shape) * 2 - 1
    test_loader = DataLoader(
        TensorDataset(clean),
        batch_size=2,
        shuffle=False,
    )

    test_mse = evaluate_checkpoint(
        device,
        model,
        test_loader,
        schedule,
        checkpoint_path,
    )

    assert isinstance(test_mse, float)
    assert torch.isfinite(torch.tensor(test_mse))

@pytest.mark.parametrize("model_name", ["cnn", "unet"])
def test_reverse_diffusion_returns_finite_image(model_name):
    schedule = make_schedule()
    model, input_shape = make_model(model_name)

    clean = torch.rand(*input_shape) * 2 - 1
    final_timestep = torch.full(
        (clean.shape[0],),
        schedule.timesteps - 1,
        dtype=torch.long,
    )
    forward_noise = torch.randn_like(clean)
    noisy = schedule.add_noise(clean, final_timestep, forward_noise)

    reconstruction = reverse_diffusion_smoke_test(model, noisy, schedule)

    assert reconstruction.shape == clean.shape
    assert torch.isfinite(reconstruction).all()
    assert torch.all(reconstruction >= -1.0)
    assert torch.all(reconstruction <= 1.0)