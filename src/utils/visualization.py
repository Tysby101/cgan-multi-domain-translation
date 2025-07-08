import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import seaborn as sns
import os


def tensor_to_image(tensor):
    """Convert tensor to numpy image for visualization"""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image in batch
    
    # Convert from [-1, 1] to [0, 1]
    img = (tensor + 1) / 2
    img = torch.clamp(img, 0, 1)
    
    # Convert to numpy and transpose
    img = img.cpu().numpy().transpose(1, 2, 0)
    
    return img


def save_sample_images(source, target, fake, save_path, num_images=4):
    """Save sample images in a grid format"""
    batch_size = min(num_images, source.size(0))
    
    # Create figure
    fig, axes = plt.subplots(3, batch_size, figsize=(4 * batch_size, 12))
    
    if batch_size == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(batch_size):
        # Source image
        source_img = tensor_to_image(source[i])
        axes[0, i].imshow(source_img)
        axes[0, i].set_title(f'Source {i+1}')
        axes[0, i].axis('off')
        
        # Target image
        target_img = tensor_to_image(target[i])
        axes[1, i].imshow(target_img)
        axes[1, i].set_title(f'Target {i+1}')
        axes[1, i].axis('off')
        
        # Generated image
        fake_img = tensor_to_image(fake[i])
        axes[2, i].imshow(fake_img)
        axes[2, i].set_title(f'Generated {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_grid(images_dict, save_path, titles=None, max_images=8):
    """
    Create a comparison grid from multiple image sets
    
    Args:
        images_dict: Dictionary with keys as row labels and values as image tensors
        save_path: Path to save the grid
        titles: Optional list of column titles
        max_images: Maximum number of images per row
    """
    num_rows = len(images_dict)
    num_cols = min(max_images, list(images_dict.values())[0].size(0))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, (label, images) in enumerate(images_dict.items()):
        for col_idx in range(num_cols):
            img = tensor_to_image(images[col_idx])
            axes[row_idx, col_idx].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator losses
    axes[0, 0].plot(epochs, [loss['g_total'] for loss in train_losses], label='Train', color='blue')
    if val_losses:
        axes[0, 0].plot(epochs, [loss.get('g_total', 0) for loss in val_losses], label='Val', color='red')
    axes[0, 0].set_title('Generator Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Discriminator losses
    axes[0, 1].plot(epochs, [loss['d_total'] for loss in train_losses], label='Train', color='blue')
    axes[0, 1].set_title('Discriminator Total Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # L1 Loss
    axes[1, 0].plot(epochs, [loss['g_l1'] for loss in train_losses], label='Train L1', color='green')
    axes[1, 0].set_title('L1 Reconstruction Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Perceptual Loss
    axes[1, 1].plot(epochs, [loss['g_perceptual'] for loss in train_losses], label='Train Perceptual', color='purple')
    axes[1, 1].set_title('Perceptual Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path, title="Metrics Comparison"):
    """Plot comparison of different metrics"""
    metrics_names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(metrics_names, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    ax.set_title(title)
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_attention_visualization(model, input_tensor, save_path):
    """Visualize attention maps from the generator"""
    model.eval()
    
    # Extract attention maps during forward pass
    attention_maps = []
    
    def hook_fn(module, input, output):
        if hasattr(module, 'gamma'):  # Self-attention module
            # Get attention weights
            batch_size, channels, height, width = input[0].size()
            proj_query = module.query(input[0]).view(batch_size, -1, height * width).permute(0, 2, 1)
            proj_key = module.key(input[0]).view(batch_size, -1, height * width)
            
            attention = torch.bmm(proj_query, proj_key)
            attention = torch.softmax(attention, dim=-1)
            
            # Reshape and store
            attention_map = attention[0].mean(0).view(height, width).detach().cpu().numpy()
            attention_maps.append(attention_map)
    
    # Register hooks for attention layers
    hooks = []
    for module in model.modules():
        if hasattr(module, 'gamma'):  # Self-attention module
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot attention maps
    if attention_maps:
        num_maps = len(attention_maps)
        fig, axes = plt.subplots(1, num_maps + 1, figsize=(4 * (num_maps + 1), 4))
        
        # Original image
        orig_img = tensor_to_image(input_tensor[0])
        axes[0].imshow(orig_img)
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        # Attention maps
        for i, attention_map in enumerate(attention_maps):
            im = axes[i + 1].imshow(attention_map, cmap='hot', interpolation='bilinear')
            axes[i + 1].set_title(f'Attention {i + 1}')
            axes[i + 1].axis('off')
            plt.colorbar(im, ax=axes[i + 1])
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_feature_visualization(model, input_tensor, save_path, layer_names=None):
    """Visualize intermediate feature maps"""
    model.eval()
    
    features = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    if layer_names is None:
        layer_names = ['encoder2', 'encoder4', 'decoder2', 'decoder4']
    
    for name, module in model.named_modules():
        if any(layer_name in name for layer_name in layer_names):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize features
    if features:
        num_layers = len(features)
        fig, axes = plt.subplots(2, num_layers, figsize=(4 * num_layers, 8))
        
        if num_layers == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (name, feature_map) in enumerate(features.items()):
            # Take first channel of first image in batch
            feature = feature_map[0, 0].cpu().numpy()
            
            # Original feature map
            axes[0, i].imshow(feature, cmap='viridis')
            axes[0, i].set_title(f'{name} - Channel 0')
            axes[0, i].axis('off')
            
            # Feature statistics
            mean_feature = feature_map[0].mean(0).cpu().numpy()
            axes[1, i].imshow(mean_feature, cmap='viridis')
            axes[1, i].set_title(f'{name} - Mean')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_loss_distribution(losses_dict, save_path):
    """Plot distribution of losses"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, (loss_name, loss_values) in enumerate(losses_dict.items()):
        if i >= 4:  # Only plot first 4 loss types
            break
            
        axes[i].hist(loss_values, bins=50, alpha=0.7, color=colors[i])
        axes[i].set_title(f'{loss_name} Distribution')
        axes[i].set_xlabel('Loss Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(loss_values)
        std_val = np.std(loss_values)
        axes[i].axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.4f}')
        axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', 
                       label=f'±1σ: {std_val:.4f}')
        axes[i].axvline(mean_val - std_val, color='orange', linestyle='--')
        axes[i].legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_progress_animation(image_lists, save_path, duration=0.5):
    """Create GIF showing training progress"""
    from PIL import Image as PILImage
    
    frames = []
    
    for epoch_images in image_lists:
        # Convert tensor to PIL Image
        if isinstance(epoch_images, torch.Tensor):
            img_array = tensor_to_image(epoch_images)
            img_array = (img_array * 255).astype(np.uint8)
            frame = PILImage.fromarray(img_array)
        else:
            frame = epoch_images
        
        frames.append(frame)
    
    # Save as GIF
    if frames:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(duration * 1000),
            loop=0
        )


def plot_model_architecture(model, input_size=(3, 256, 256), save_path=None):
    """Plot model architecture diagram"""
    try:
        from torchviz import make_dot
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_size)
        
        # Forward pass
        output = model(dummy_input)
        
        # Create computation graph
        dot = make_dot(output, params=dict(model.named_parameters()))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            dot.render(save_path.replace('.png', ''), format='png')
        
        return dot
        
    except ImportError:
        print("torchviz not available. Install with: pip install torchviz")
        return None


def test_visualization():
    """Test visualization functions"""
    # Create dummy data
    batch_size = 4
    source = torch.randn(batch_size, 3, 256, 256)
    target = torch.randn(batch_size, 3, 256, 256)
    fake = torch.randn(batch_size, 3, 256, 256)
    
    # Test save_sample_images
    save_sample_images(source, target, fake, './test_samples.png')
    print("Sample images saved to: ./test_samples.png")
    
    # Test comparison grid
    images_dict = {
        'Source': source,
        'Target': target,
        'Generated': fake
    }
    create_comparison_grid(images_dict, './test_grid.png')
    print("Comparison grid saved to: ./test_grid.png")
    
    # Test metrics plotting
    metrics = {'SSIM': 0.85, 'PSNR': 28.5, 'LPIPS': 0.12, 'FID': 15.3}
    plot_metrics_comparison(metrics, './test_metrics.png')
    print("Metrics plot saved to: ./test_metrics.png")

