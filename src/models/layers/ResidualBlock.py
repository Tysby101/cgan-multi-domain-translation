import torch.nn as nn
from torch.nn.utils import spectral_norm
from src.models.layers import SelfAttention


class ResidualBlock(nn.Module):
    """Residual block with optional attention"""
    def __init__(self, channels, use_attention=False, use_spectral_norm=True):
        super().__init__()
        
        conv_layer = spectral_norm if use_spectral_norm else lambda x: x
        
        self.conv1 = conv_layer(nn.Conv2d(channels, channels, 3, padding=1))
        self.conv2 = conv_layer(nn.Conv2d(channels, channels, 3, padding=1))
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        
        self.attention = SelfAttention(channels) if use_attention else None
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        if self.attention is not None:
            out = self.attention(out)
            
        return F.relu(out + residual)
