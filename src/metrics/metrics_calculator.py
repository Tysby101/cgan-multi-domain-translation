import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from src.metrics.perceptual_distance import PerceptualDistance

class MetricsCalculator:
    """Class to calculate various image quality metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        
        # Freeze LPIPS model
        for param in self.lpips_model.parameters():
            param.requires_grad = False
    
    def calculate_ssim(self, img1, img2):
        """
        Calculate SSIM between two images
        
        Args:
            img1, img2: Tensors of shape (B, C, H, W) in range [-1, 1]
        
        Returns:
            SSIM score
        """
        # Convert to numpy and rescale to [0, 1]
        img1_np = ((img1.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1)
        img2_np = ((img2.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1)
        
        ssim_scores = []
        for i in range(img1_np.shape[0]):
            score = ssim(img1_np[i], img2_np[i], 
                        multichannel=True, channel_axis=-1, data_range=1.0)
            ssim_scores.append(score)
        
        return np.mean(ssim_scores)
    
    def calculate_psnr(self, img1, img2):
        """
        Calculate PSNR between two images
        
        Args:
            img1, img2: Tensors of shape (B, C, H, W) in range [-1, 1]
        
        Returns:
            PSNR score
        """
        # Convert to numpy and rescale to [0, 1]
        img1_np = ((img1.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1)
        img2_np = ((img2.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1)
        
        psnr_scores = []
        for i in range(img1_np.shape[0]):
            score = psnr(img1_np[i], img2_np[i], data_range=1.0)
            psnr_scores.append(score)
        
        return np.mean(psnr_scores)
    
    def calculate_lpips(self, img1, img2):
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity)
        
        Args:
            img1, img2: Tensors of shape (B, C, H, W) in range [-1, 1]
        
        Returns:
            LPIPS score (lower is better)
        """
        with torch.no_grad():
            score = self.lpips_model(img1, img2)
        return score.mean().item()
    
    def calculate_mse(self, img1, img2):
        """Calculate Mean Squared Error"""
        mse = torch.mean((img1 - img2) ** 2)
        return mse.item()
    
    def calculate_mae(self, img1, img2):
        """Calculate Mean Absolute Error"""
        mae = torch.mean(torch.abs(img1 - img2))
        return mae.item()
    
    def calculate_all_metrics(self, pred, target):
        """
        Calculate all metrics between predicted and target images
        
        Args:
            pred, target: Tensors of shape (B, C, H, W) in range [-1, 1]
        
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        with torch.no_grad():
            metrics['ssim'] = self.calculate_ssim(pred, target)
            metrics['psnr'] = self.calculate_psnr(pred, target)
            metrics['lpips'] = self.calculate_lpips(pred, target)
            metrics['mse'] = self.calculate_mse(pred, target)
            metrics['mae'] = self.calculate_mae(pred, target)
        
        return metrics
    

def batch_calculate_metrics(generator, dataloader, device, save_path=None):
    """
    Calculate metrics for an entire dataset
    
    Args:
        generator: Trained generator model
        dataloader: DataLoader containing source-target pairs
        device: Device to use
        save_path: Optional path to save generated images
    
    Returns:
        Dictionary of average metrics
    """
    generator.eval()
    metrics_calc = MetricsCalculator(device)
    
    all_metrics = {
        'ssim': [],
        'psnr': [],
        'lpips': [],
        'mse': [],
        'mae': []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            # Generate fake images
            fake = generator(source)
            
            # Calculate metrics
            batch_metrics = metrics_calc.calculate_all_metrics(fake, target)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                all_metrics[key].append(value)
            
            # Optionally save images
            if save_path and batch_idx < 10:  # Save first 10 batches
                save_comparison_images(
                    source, target, fake,
                    f"{save_path}/comparison_batch_{batch_idx}.png"
                )
    
    # Calculate averages
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return avg_metrics


def save_comparison_images(source, target, fake, save_path):
    """Save comparison images side by side"""
    
    
    # Convert tensors to numpy arrays
    def tensor_to_numpy(tensor):
        return ((tensor.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1)
    
    source_np = tensor_to_numpy(source)
    target_np = tensor_to_numpy(target)
    fake_np = tensor_to_numpy(fake)
    
    # Create comparison plot
    batch_size = min(4, source.size(0))
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        axes[i, 0].imshow(source_np[i])
        axes[i, 0].set_title('Source')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target_np[i])
        axes[i, 1].set_title('Target')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(fake_np[i])
        axes[i, 2].set_title('Generated')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_metrics():
    """Test function for metrics calculation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy data
    img1 = torch.randn(4, 3, 256, 256).to(device)
    img2 = torch.randn(4, 3, 256, 256).to(device)
    
    # Test metrics calculator
    metrics_calc = MetricsCalculator(device)
    metrics = metrics_calc.calculate_all_metrics(img1, img2)
    
    print("Calculated metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test perceptual distance
    perc_dist = PerceptualDistance('alex', device)
    distance = perc_dist.calculate_distance(img1, img2)
    print(f"  Perceptual distance (Alex): {distance:.4f}")


if __name__ == "__main__":
    test_metrics()