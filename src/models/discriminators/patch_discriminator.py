import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for single scale"""
    def __init__(self, input_channels=6, features=64, use_spectral_norm=True):
        super().__init__()
        
        conv_layer = spectral_norm if use_spectral_norm else lambda x: x
        
        # No normalization for first layer
        self.layer1 = conv_layer(nn.Conv2d(input_channels, features, 4, 2, 1))
        
        # Subsequent layers with normalization
        self.layer2 = conv_layer(nn.Conv2d(features, features * 2, 4, 2, 1))
        self.norm2 = nn.InstanceNorm2d(features * 2)
        
        self.layer3 = conv_layer(nn.Conv2d(features * 2, features * 4, 4, 2, 1))
        self.norm3 = nn.InstanceNorm2d(features * 4)
        
        self.layer4 = conv_layer(nn.Conv2d(features * 4, features * 8, 4, 1, 1))
        self.norm4 = nn.InstanceNorm2d(features * 8)
        
        # Final layer - no normalization
        self.layer5 = conv_layer(nn.Conv2d(features * 8, 1, 4, 1, 1))
        
    def forward(self, x):
        # Extract features for feature matching loss
        features = []
        
        x = F.leaky_relu(self.layer1(x), 0.2)
        features.append(x)
        
        x = F.leaky_relu(self.norm2(self.layer2(x)), 0.2)
        features.append(x)
        
        x = F.leaky_relu(self.norm3(self.layer3(x)), 0.2)
        features.append(x)
        
        x = F.leaky_relu(self.norm4(self.layer4(x)), 0.2)
        features.append(x)
        
        x = self.layer5(x)
        
        return x, features
