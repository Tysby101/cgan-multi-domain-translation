import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SpectralNormDiscriminator(nn.Module):
    """Alternative discriminator with spectral normalization throughout"""
    def __init__(self, input_channels=6, features=64):
        super().__init__()
        
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, features, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(features, features * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(features * 2, features * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(features * 4, features * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(features * 8, 1, 4, 1, 1))
        )
        
    def forward(self, x):
        return self.layers(x)