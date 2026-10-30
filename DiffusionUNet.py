import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Computes sinusoidal time step embeddings to inform the network 
    of the current diffusion timestep (t).
    """
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


class Block(nn.Module):
    """A single Convolution -> GroupNorm -> SiLU layer."""
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DDPMResidualBlock(nn.Module):
    """
    A full residual block that injects time embeddings into the feature maps.
    Uses GroupNorm and SiLU activation.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, num_groups=8):
        super().__init__()
        self.block1 = Block(in_ch, out_ch, num_groups)
        self.block2 = Block(out_ch, out_ch, num_groups)
        
        # Project time embedding to match output channel dimension
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        # Shortcut connection if channels change
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        # Add time embedding (expanding dimensions from [B, C] to [B, C, 1, 1])
        time_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    """
    Standard multi-head linear self-attention layer applied at lower spatial resolutions.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.num_heads, c // self.num_heads, h * w).transpose(-1, -2), qkv)

        # Attention mechanism
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.to_out(out)


class DDPMUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # Time Embedding Multi-Layer Perceptron
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Contracting Path (Encoder)
        self.inc = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.down1 = DDPMResidualBlock(base_channels, base_channels, time_emb_dim)
        self.pool1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1) # Strided conv downsampling
        
        self.down2 = DDPMResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        # Bottleneck (Middle)
        self.mid1 = DDPMResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.attn = SelfAttention(base_channels * 4)
        self.mid2 = DDPMResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        
        # Expansive Path (Decoder)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.conv_up1 = DDPMResidualBlock(base_channels * 4, base_channels, time_emb_dim) # Skip connection doubles channels
        
        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.conv_up2 = DDPMResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        
        # Final Projection
        self.outc = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        # 1. Embed the timestep
        t_emb = self.time_mlp(t)
        
        # 2. Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.pool1(x2)
        x4 = self.down2(x3, t_emb)
        x5 = self.pool2(x4)
        
        # 3. Bottleneck with Self-Attention
        x5 = self.mid1(x5, t_emb)
        x5 = self.attn(x5)
        x5 = self.mid2(x5, t_emb)
        
        # 4. Decoder with Skip Connections
        x_up = self.up1(x5)
        x_up = torch.cat([x_up, x4], dim=1) # Cat along channel dim
        x_up = self.conv_up1(x_up, t_emb)
        
        x_up = self.up2(x_up)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.conv_up2(x_up, t_emb)
        
        return self.outc(x_up)