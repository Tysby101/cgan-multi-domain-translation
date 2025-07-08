#!/usr/bin/env python3
"""
Training script for Multi-Domain cGAN
"""

import os
import sys
import argparse
import yaml
import torch
import wandb
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.datasets import create_dataloaders
from training.trainer import CGANTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Multi-Domain cGAN')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    parser.add_argument('--wandb-project', type=str, default='multi-domain-cgan',
                       help='W&B project name')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='W&B run name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(gpu_id):
    """Setup device for training"""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def initialize_wandb(config, args):
    """Initialize Weights & Biases logging"""
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config,
        save_code=True
    )
    
    # Log code files
    wandb.run.log_code(".")


def validate_config(config):
    """Validate configuration parameters"""
    required_keys = [
        'model', 'training', 'data', 'paths'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Check data paths
    data_root = config['paths']['data_root']
    if not os.path.exists(data_root):
        raise ValueError(f"Data root path does not exist: {data_root}")
    
    print("Configuration validation passed")


def create_directories(config):
    """Create necessary directories"""
    directories = [
        config['paths']['checkpoints'],
        config['paths']['logs'],
        config['paths']['results']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Validate configuration
    validate_config(config)
    
    # Set random seed
    set_seed(args.seed)
    print(f"Set random seed to: {args.seed}")
    
    # Setup device
    device = setup_device(args.gpu)
    config['device'] = device
    
    # Create directories
    create_directories(config)
    
    # Initialize W&B
    initialize_wandb(config, args)
    
    try:
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_dataloaders(config)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Initialize trainer
        print("Initializing trainer...")
        trainer = CGANTrainer(config, train_loader, val_loader, device)
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        print("Starting training...")
        trainer.train()
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
        # Save emergency checkpoint
        emergency_path = os.path.join(config['paths']['checkpoints'], 'emergency.pth')
        if 'trainer' in locals():
            trainer.save_checkpoint()
            print(f"Emergency checkpoint saved to: {emergency_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        # Finish W&B run
        wandb.finish()


if __name__ == "__main__":
    main()