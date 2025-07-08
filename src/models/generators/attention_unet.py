import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.models.layers.SelfAttention import SelfAttention
from src.models.layers.ResidualBlock import ResidualBlock


class AttentionUNet(nn.Module):
    """U-Net Generator with Self-Attention"""
    def __init__(self, input_channels=3, output_channels=3, features=64, 
                 attention_layers=[2, 4, 6], use_spectral_norm=True):
        super().__init__()
        
        conv_layer = spectral_norm if use_spectral_norm else lambda x: x
        
        # Encoder
        self.encoder1 = conv_layer(nn.Conv2d(input_channels, features, 4, 2, 1))
        self.encoder2 = conv_layer(nn.Conv2d(features, features * 2, 4, 2, 1))
        self.encoder3 = conv_layer(nn.Conv2d(features * 2, features * 4, 4, 2, 1))
        self.encoder4 = conv_layer(nn.Conv2d(features * 4, features * 8, 4, 2, 1))
        self.encoder5 = conv_layer(nn.Conv2d(features * 8, features * 8, 4, 2, 1))
        self.encoder6 = conv_layer(nn.Conv2d(features * 8, features * 8, 4, 2, 1))
        
        # Bottleneck with attention
        self.bottleneck = ResidualBlock(features * 8, use_attention=True, 
                                      use_spectral_norm=use_spectral_norm)
        
        # Decoder
        self.decoder1 = conv_layer(nn.ConvTranspose2d(features * 8, features * 8, 4, 2, 1))
        self.decoder2 = conv_layer(nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1))
        self.decoder3 = conv_layer(nn.ConvTranspose2d(features * 16, features * 4, 4, 2, 1))
        self.decoder4 = conv_layer(nn.ConvTranspose2d(features * 8, features * 2, 4, 2, 1))
        self.decoder5 = conv_layer(nn.ConvTranspose2d(features * 4, features, 4, 2, 1))
        self.decoder6 = conv_layer(nn.ConvTranspose2d(features * 2, output_channels, 4, 2, 1))
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            SelfAttention(features * (2 ** i)) 
            for i in attention_layers
        ])
        
        # Normalization layers
        self.norm_layers = nn.ModuleList([
            nn.InstanceNorm2d(features * (2 ** i)) 
            for i in range(6)
        ])
        
    def forward(self, x):
        # Encoder
        e1 = F.leaky_relu(self.encoder1(x), 0.2)
        e2 = F.leaky_relu(self.norm_layers[1](self.encoder2(e1)), 0.2)
        e3 = F.leaky_relu(self.norm_layers[2](self.encoder3(e2)), 0.2)
        e4 = F.leaky_relu(self.norm_layers[3](self.encoder4(e3)), 0.2)
        e5 = F.leaky_relu(self.norm_layers[4](self.encoder5(e4)), 0.2)
        e6 = F.leaky_relu(self.norm_layers[5](self.encoder6(e5)), 0.2)
        
        # Bottleneck
        b = self.bottleneck(e6)
        
        # Decoder with skip connections
        d1 = F.relu(self.norm_layers[5](self.decoder1(b)))
        d1 = torch.cat([d1, e5], 1)
        
        d2 = F.relu(self.norm_layers[4](self.decoder2(d1)))
        d2 = torch.cat([d2, e4], 1)
        
        d3 = F.relu(self.norm_layers[3](self.decoder3(d2)))
        d3 = torch.cat([d3, e3], 1)
        
        d4 = F.relu(self.norm_layers[2](self.decoder4(d3)))
        d4 = torch.cat([d4, e2], 1)
        
        d5 = F.relu(self.norm_layers[1](self.decoder5(d4)))
        d5 = torch.cat([d5, e1], 1)
        
        d6 = self.decoder6(d5)
        
        return torch.tanh(d6)


def test_generator():
    """Test function for the generator"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet().to(device)
    
    # Test with random input
    x = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    test_generator()