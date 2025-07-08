import torch
import torch.nn.functional as F
from src.models.discriminators.multi_scale_discriminator import MultiScaleDiscriminator

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolation
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    
    # Get discriminator output
    if isinstance(discriminator, MultiScaleDiscriminator):
        disc_interpolates, _ = discriminator(interpolates)
        disc_interpolates = disc_interpolates[0]  # Use first scale
    else:
        disc_interpolates = discriminator(interpolates)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty