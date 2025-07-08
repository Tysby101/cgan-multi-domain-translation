import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.discriminators.patch_discriminator import PatchDiscriminator
from src.utils.utils import compute_gradient_penalty

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for better detail preservation"""
    def __init__(self, input_channels=6, features=64, num_scales=3, use_spectral_norm=True):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Create multiple discriminators at different scales
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(input_channels, features, use_spectral_norm)
            for _ in range(num_scales)
        ])
        
        # Downsampling layers for multi-scale input
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        """
        Forward pass through all scales
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            outputs: List of discriminator outputs at each scale
            features: List of feature maps at each scale
        """
        outputs = []
        all_features = []
        
        current_input = x
        
        for i, discriminator in enumerate(self.discriminators):
            output, features = discriminator(current_input)
            outputs.append(output)
            all_features.append(features)
            
            # Downsample for next scale (except for last scale)
            if i < self.num_scales - 1:
                current_input = self.downsample(current_input)
        
        return outputs, all_features
    

def test_discriminator():
    """Test function for the discriminator"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test multi-scale discriminator
    discriminator = MultiScaleDiscriminator().to(device)
    
    # Test with random input (source + target concatenated)
    x = torch.randn(2, 6, 256, 256).to(device)
    
    with torch.no_grad():
        outputs, features = discriminator(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Number of scales: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Scale {i} output shape: {output.shape}")
    
    print(f"Model parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Test gradient penalty
    real_samples = torch.randn(2, 6, 256, 256).to(device)
    fake_samples = torch.randn(2, 6, 256, 256).to(device)
    
    gp = compute_gradient_penalty(discriminator, real_samples, fake_samples, device)
    print(f"Gradient penalty: {gp.item()}")
    
    return discriminator


if __name__ == "__main__":
    test_discriminator()