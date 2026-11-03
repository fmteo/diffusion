from io import BytesIO
from pathlib import Path
from PIL import Image

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

from DiffusionCNN import SinusoidalPositionEmbeddings, SimpleCNNBlock, SimpleDDPMCNN
from DiffusionUNet import SinusoidalPositionEmbeddings, Block, DDPMResidualBlock, SelfAttention, DDPMUNet

# -------------------- App settings --------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MNIST_models = {"cnn": "models/ddpm_cnn_mnist.pt", "unet": "models/ddpm_unet_mnist.pt"}
Image_models = {"cnn": "models/ddpm_cnn_image.pt", "unet": "models/ddpm_unet_image.pt"}

# -------------------- Load checkpoint and schedule --------------------

@st.cache_resource
def load_models(cnn_path, unet_path, is_mnist):
    # Modification times are cache keys, so changed checkpoints are reloaded.
    cnn_mtime = Path(cnn_path).stat().st_mtime
    unet_mtime = Path(unet_path).stat().st_mtime
    del cnn_mtime, unet_mtime

    cnn_checkpoint = torch.load(
        cnn_path, map_location=DEVICE, weights_only=False
    )

    unet_checkpoint = torch.load(
        unet_path, map_location=DEVICE, weights_only=False
    )

    schedule_keys = ("T", "beta_start", "beta_end")
    for key in schedule_keys:
        if unet_checkpoint[key] != cnn_checkpoint[key]:
            raise ValueError(
                f"The CNN and U-Net checkpoints have different {key} values. "
                "Train them with the same diffusion settings."
            )

    channel = 1 if is_mnist else 3

    cnn = SimpleDDPMCNN(in_channels=channel, out_channels=channel,
        hidden_dim=cnn_checkpoint.get("width", 64)
    ).to(DEVICE)
    cnn.load_state_dict(cnn_checkpoint["model_state_dict"])
    cnn.eval()

    unet = DDPMUNet(in_channels=channel, out_channels=channel,
        base_channels=unet_checkpoint.get("width", 64)
    ).to(DEVICE)
    unet.load_state_dict(unet_checkpoint["model_state_dict"])
    unet.eval()

    total_steps = cnn_checkpoint["T"]
    betas = torch.linspace(
        cnn_checkpoint["beta_start"],
        cnn_checkpoint["beta_end"],
        total_steps,
        device=DEVICE,
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return cnn, unet, betas, alphas, alpha_bars, total_steps

# -------------------- Image helpers --------------------

def prepare_image(image_bytes, is_mnist):
    image = Image.open(BytesIO(image_bytes))

    image_size = 28 if is_mnist else 128

    image = image.resize(
        (image_size, image_size),
        Image.Resampling.LANCZOS,
    )
    
    if is_mnist:
        image = image.convert("L")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        image = image.convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return image, image_tensor

def tensor_to_pil(tensor):
    # 1. Detach, move to CPU, and remove batch dimensions of size 1
    tensor = tensor.detach().cpu().squeeze()
    
    # 2. Denormalize from [-1, 1] to [0, 1] and clamp to stay safe
    tensor = tensor.clamp(-1, 1).add(1).div(2)
    
    # 3. Change shape from [C, H, W] to [H, W, C] if it's a 3D tensor
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # 4. Scale to [0, 255] and convert to a NumPy uint8 array
    pixels = tensor.mul(255).byte().numpy()
    
    # 5. Determine the mode automatically based on channels
    mode = "RGB" if tensor.dim() == 3 and tensor.shape[-1] == 3 else "L"
    
    return Image.fromarray(pixels, mode=mode)

def randn_like_with_generator(tensor, generator):
    return torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype,
        generator=generator,
    )

# -------------------- Forward diffusion --------------------

@torch.no_grad()
def forward_diffusion(clean, timestep_index, alphas, generator):
    """
    Recursively add noise:
        z_t = sqrt(alpha_t) * z_(t-1)
              + sqrt(1-alpha_t) * epsilon_t

    Returns all states from the clean image through the selected noisy state.
    """
    image = clean.clone()
    stages = [image.clone()]

    for index in range(timestep_index + 1):
        alpha_t = alphas[index]
        noise = randn_like_with_generator(image, generator)

        image = (
            alpha_t.sqrt() * image
            + (1.0 - alpha_t).sqrt() * noise
        )
        stages.append(image.clone())

    return image, stages

# -------------------- Reverse DDPM --------------------

@torch.no_grad()
def reverse_ddpm(
    model,
    noisy,
    timestep_index,
    betas,
    alphas,
    alpha_bars,
    generator,
    progress_bar,
):
    """Reverse from the selected noisy timestep to the reconstructed image."""
    image = noisy.clone()

    # Each entry is (display timestep, image tensor).
    stages = [(timestep_index + 1, image.clone())]

    for index in range(timestep_index, -1, -1):
        timestep_tensor = torch.tensor(
            [index], device=DEVICE, dtype=torch.long
        )

        beta_t = betas[index]
        alpha_t = alphas[index]
        alpha_bar_t = alpha_bars[index]

        if index == 0:
            alpha_bar_previous = torch.tensor(1.0, device=DEVICE)
        else:
            alpha_bar_previous = alpha_bars[index - 1]

        predicted_noise = model(image, timestep_tensor)

        # Estimate the clean image from z_t and predicted total noise.
        predicted_clean = (
            image - (1.0 - alpha_bar_t).sqrt() * predicted_noise
        ) / alpha_bar_t.sqrt()
        predicted_clean = predicted_clean.clamp(-1, 1)

        # DDPM posterior mean.
        coefficient_clean = (
            alpha_bar_previous.sqrt()
            * beta_t
            / (1.0 - alpha_bar_t)
        )
        coefficient_noisy = (
            alpha_t.sqrt()
            * (1.0 - alpha_bar_previous)
            / (1.0 - alpha_bar_t)
        )
        mean = (
            coefficient_clean * predicted_clean
            + coefficient_noisy * image
        )
        
        image = mean
        image = image.clamp(-1, 1)
        stages.append((index, image.clone()))

        progress_bar.progress(
            (timestep_index - index + 1) / (timestep_index + 1)
        )

    return image, stages


# -------------------- Plot helpers --------------------

def evenly_spaced_positions(length, count=5):
    """Choose positions including the first and last stage."""
    if length <= 1:
        return [0] * count

    return [
        round(i * (length - 1) / (count - 1))
        for i in range(count)
    ]

# -------------------- Streamlit app --------------------

def main():
    st.set_page_config(
        page_title="DDPMs",
        layout="wide",
    )

    st.markdown(
    """
    <style>
        .block-container {
            padding: 1rem 3rem;
        }
        .st-emotion-cache-10p9htt {
            height: 2rem;
            margin-bottom: 0;
        }
        textarea {
            resize: none !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("Denoising Diffusion Probabilistic Models (DDPMs)")

    # Settings and image upload in the narrow sidebar.
    st.sidebar.header("Settings")

     # Create the tabs by passing a list of labels
    mnist_tab, image_tab = st.sidebar.tabs(["MNIST", "Image"], on_change="rerun")

    is_mnist = True

    if mnist_tab.open:
        is_mnist = True
        cnn_path = MNIST_models["cnn"]
        unet_path =  MNIST_models["unet"]

    if image_tab.open:
        is_mnist = False
        cnn_path = Image_models["cnn"]
        unet_path =  Image_models["unet"]

    st.sidebar.text_area(label="Model(s) loaded", value=f"{cnn_path}\n{unet_path}", height=50)

    try:
        cnn_file = Path(cnn_path)
        unet_file = Path(unet_path)

        if not unet_file.is_file() or not cnn_file.is_file():
            st.sidebar.info("Enter the paths to both trained .pth checkpoints.")
            st.stop()

        model_cnn, model_unet, betas, alphas, alpha_bars, total_steps = load_models(
            str(cnn_file),
            str(unet_file),
            is_mnist
        )
    except Exception as error:
        st.sidebar.error(f"Could not load both checkpoints: {error}")
        st.stop()

    uploaded_text = "Upload a 28x28 MNIST digit image" if is_mnist else "Upload a 128x128 image"
    uploaded = st.sidebar.file_uploader(
        uploaded_text,
        key=f"file_{is_mnist}",
        type=["png", "jpg", "jpeg", "bmp"]
    )

    if uploaded is None:
        st.info("Upload an image in the sidebar to begin.")
        st.stop()

    display_input, clean = prepare_image(uploaded.getvalue(), is_mnist)
    st.sidebar.image(display_input, caption="", width=64)

    default_timestep = 350 if is_mnist else 200
    timestep = st.sidebar.slider(
        "Forward noise timestep T",
        min_value=1,
        max_value=total_steps,
        value=min(default_timestep, total_steps),
        help="Higher values add more noise before reconstruction.",
    )

    seed = 123
    run_button = st.sidebar.button(
        "Add noise and reconstruct",
        type="primary",
        use_container_width=True,
    )

    # Results occupy the wider main area.
    if not run_button:
        st.info(
            "Choose the timestep and sampling settings in the sidebar, "
            "then click **Add noise and reconstruct**."
        )
        st.stop()

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int(seed))

    timestep_index = timestep - 1

    noisy, forward_stages = forward_diffusion(
        clean,
        timestep_index,
        alphas,
        generator,
    )

    progress_cnn = st.progress(0.0, text="Reverse denoising with CNN...")
    cnn_result, cnn_stages = reverse_ddpm(
        model=model_cnn,
        noisy=noisy,
        timestep_index=timestep_index,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        generator=generator,
        progress_bar=progress_cnn,
    )
    progress_cnn.empty()

    progress_unet = st.progress(0.0, text="Reverse denoising with U-Net...")
    unet_result, unet_stages = reverse_ddpm(
        model=model_unet,
        noisy=noisy,
        timestep_index=timestep_index,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        generator=generator,
        progress_bar=progress_unet,
    )
    progress_unet.empty()

    noisy_mse = F.mse_loss(noisy, clean).item()
    cnn_mse = F.mse_loss(cnn_result, clean).item()
    unet_mse = F.mse_loss(unet_result, clean).item()
    cumulative_alpha = alpha_bars[timestep_index].item()

    forward_positions = evenly_spaced_positions(len(forward_stages), count=5)
    cnn_positions = evenly_spaced_positions(len(cnn_stages), count=5)
    unet_positions = evenly_spaced_positions(len(unet_stages), count=5)

    # Each row has one summary column and five image columns.
    row_columns = [1.5, 1, 1, 1, 1, 1]

    # Row 1: Forward diffusion.
    cols = st.columns(row_columns, gap="small")

    with cols[0]:
        #st.markdown("### Forward diffusion")
        info1, info2 = st.columns([1,2])
        info1.metric(label="**T:**", value=timestep)
        info2.metric(label="**ᾱₜ:**", value=f"{alpha_bars[timestep - 1].item():.5f}")
        st.metric(label="**Noisy-image MSE:**", value=f"{noisy_mse:.5f}")

    for image_col, position in zip(cols[1:], forward_positions):
        stage_t = position
        stage = forward_stages[position]
        image_col.image(
            tensor_to_pil(stage),
            caption=f"Forward t={stage_t}\n\nᾱ={alpha_bars[stage_t].item():.3f}",
            use_container_width=True
        )

    # Row 2: CNN reverse diffusion.
    cols = st.columns(row_columns, gap="small")

    with cols[0]:
        st.markdown("### CNN")
        st.metric(label="**MSE:**", value=f"{cnn_mse:.5f}")

    for image_col, position in zip(cols[1:], reversed(cnn_positions)):
        stage_t, stage_image = cnn_stages[position]
        image_col.image(
            tensor_to_pil(stage_image),
            caption=f"Reverse t={stage_t}\n\nMSE={F.mse_loss(stage_image, clean).item():.5f}",
            use_container_width=True
        )

    # Row 3: U-Net reverse diffusion.
    cols = st.columns(row_columns, gap="small")

    with cols[0]:
        st.markdown("### U-Net")
        st.metric(label="**MSE:**", value=f"{unet_mse:.5f}")

    for image_col, position in zip(cols[1:], reversed(unet_positions)):
        stage_t, stage_image = unet_stages[position]
        image_col.image(
            tensor_to_pil(stage_image),
            caption=f"Reverse t={stage_t}\n\nMSE={F.mse_loss(stage_image, clean).item():.5f}",
            use_container_width=True
        )

if __name__ == "__main__":
    main()