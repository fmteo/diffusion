import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    """Computes time step embeddings to inform the network of the diffusion step (t)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleCNNBlock(nn.Module):
    """A straightforward CNN layer that blends image features with time embeddings."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        # Standard convolutions keeping spatial size exactly the same (padding=1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        
        # Project time embedding to match the channel count
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        # Shortcut connection if channel dimensions don't match
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # First convolution
        h = self.act(self.norm1(self.conv1(x)))
        
        # Inject time embedding [B, C] -> [B, C, 1, 1]
        time_feature = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_feature
        
        # Second convolution
        h = self.act(self.norm2(self.conv2(h)))
        
        # Residual skip connection
        return h + self.shortcut(x)


class SimpleDDPMCNN(nn.Module):
    """
    Maintains the image's original resolution all the way through.
    """
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64):
        super().__init__()
        
        # 1. Time Projection MLP
        time_emb_dim = hidden_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 2. Flat CNN Layers (No Pooling, No Upsampling)
        self.init_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        self.layer1 = SimpleCNNBlock(hidden_dim, hidden_dim, time_emb_dim)
        self.layer2 = SimpleCNNBlock(hidden_dim, hidden_dim * 2, time_emb_dim)
        self.layer3 = SimpleCNNBlock(hidden_dim * 2, hidden_dim, time_emb_dim)
        
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # Embed the timestep
        t_emb = self.time_mlp(t)
        
        # Process through the simple CNN chain
        out = self.init_conv(x)
        out = self.layer1(out, t_emb)
        out = self.layer2(out, t_emb)
        out = self.layer3(out, t_emb)
        
        return self.final_conv(out)