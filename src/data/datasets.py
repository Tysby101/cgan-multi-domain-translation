import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from typing import Tuple, List, Optional


class CelebAFFHQDataset(Dataset):
    """Dataset for CelebA-HQ and FFHQ image-to-image translation"""
    
    def __init__(self, root_dir: str, transform=None, mode='train', 
                 task='face_enhancement', split_ratio=0.8):
        """
        Args:
            root_dir: Root directory containing the datasets
            transform: Optional transform to be applied
            mode: 'train', 'val', or 'test'
            task: Type of translation task
            split_ratio: Ratio for train/val/test split
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.task = task
        
        # Define paths
        self.celeba_path = os.path.join(root_dir, 'CelebA-HQ')
        self.ffhq_path = os.path.join(root_dir, 'FFHQ')
        
        # Load image pairs based on task
        self.image_pairs = self._load_image_pairs()
        
        # Split dataset
        self._split_dataset(split_ratio)
    
    def _load_image_pairs(self) -> List[Tuple[str, str]]:
        """Load image pairs based on the task"""
        pairs = []
        
        if self.task == 'face_enhancement':
            # CelebA-HQ (input) -> FFHQ (target)
            celeba_images = sorted([f for f in os.listdir(self.celeba_path) 
                                  if f.endswith(('.jpg', '.png'))])
            ffhq_images = sorted([f for f in os.listdir(self.ffhq_path) 
                                if f.endswith(('.jpg', '.png'))])
            
            # Create pairs (assuming same number of images)
            min_length = min(len(celeba_images), len(ffhq_images))
            for i in range(min_length):
                pairs.append((
                    os.path.join(self.celeba_path, celeba_images[i]),
                    os.path.join(self.ffhq_path, ffhq_images[i])
                ))
        
        elif self.task == 'style_transfer':
            # Random pairing for style transfer
            celeba_images = [os.path.join(self.celeba_path, f) 
                           for f in os.listdir(self.celeba_path) 
                           if f.endswith(('.jpg', '.png'))]
            ffhq_images = [os.path.join(self.ffhq_path, f) 
                         for f in os.listdir(self.ffhq_path) 
                         if f.endswith(('.jpg', '.png'))]
            
            # Create random pairs
            min_length = min(len(celeba_images), len(ffhq_images))
            random.shuffle(ffhq_images)
            for i in range(min_length):
                pairs.append((celeba_images[i], ffhq_images[i]))
        
        return pairs
    
    def _split_dataset(self, split_ratio):
        """Split dataset into train/val/test"""
        total_length = len(self.image_pairs)
        train_length = int(total_length * split_ratio)
        val_length = int(total_length * 0.1)
        
        if self.mode == 'train':
            self.image_pairs = self.image_pairs[:train_length]
        elif self.mode == 'val':
            self.image_pairs = self.image_pairs[train_length:train_length + val_length]
        else:  # test
            self.image_pairs = self.image_pairs[train_length + val_length:]
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        source_path, target_path = self.image_pairs[idx]
        
        # Load images
        source_image = Image.open(source_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        
        return {
            'source': source_image,
            'target': target_image,
            'source_path': source_path,
            'target_path': target_path
        }


class PairedImageDataset(Dataset):
    """Generic paired image dataset"""
    
    def __init__(self, source_dir: str, target_dir: str, transform=None, 
                 mode='train', extensions=('.jpg', '.png')):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        self.mode = mode
        
        # Get image filenames
        self.source_images = sorted([f for f in os.listdir(source_dir) 
                                   if f.endswith(extensions)])
        self.target_images = sorted([f for f in os.listdir(target_dir) 
                                   if f.endswith(extensions)])
        
        # Ensure equal number of images
        assert len(self.source_images) == len(self.target_images), \
            "Source and target directories must contain equal number of images"
    
    def __len__(self):
        return len(self.source_images)
    
    def __getitem__(self, idx):
        source_path = os.path.join(self.source_dir, self.source_images[idx])
        target_path = os.path.join(self.target_dir, self.target_images[idx])
        
        source_image = Image.open(source_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')
        
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        
        return {
            'source': source_image,
            'target': target_image,
            'source_path': source_path,
            'target_path': target_path
        }


class UnpairedImageDataset(Dataset):
    """Dataset for unpaired image translation (CycleGAN style)"""
    
    def __init__(self, domain_a_dir: str, domain_b_dir: str, transform=None, 
                 mode='train', extensions=('.jpg', '.png')):
        self.domain_a_dir = domain_a_dir
        self.domain_b_dir = domain_b_dir
        self.transform = transform
        self.mode = mode
        
        # Get image filenames
        self.domain_a_images = sorted([f for f in os.listdir(domain_a_dir) 
                                     if f.endswith(extensions)])
        self.domain_b_images = sorted([f for f in os.listdir(domain_b_dir) 
                                     if f.endswith(extensions)])
    
    def __len__(self):
        return max(len(self.domain_a_images), len(self.domain_b_images))
    
    def __getitem__(self, idx):
        # Wrap around if one domain has fewer images
        a_idx = idx % len(self.domain_a_images)
        b_idx = idx % len(self.domain_b_images)
        
        a_path = os.path.join(self.domain_a_dir, self.domain_a_images[a_idx])
        b_path = os.path.join(self.domain_b_dir, self.domain_b_images[b_idx])
        
        image_a = Image.open(a_path).convert('RGB')
        image_b = Image.open(b_path).convert('RGB')
        
        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
        
        return {
            'domain_a': image_a,
            'domain_b': image_b,
            'a_path': a_path,
            'b_path': b_path
        }


def get_transforms(image_size=256, mode='train'):
    """Get data transforms for training and validation"""
    
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size + 30, image_size + 30)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                                 saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    return transform


def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    
    # Get transforms
    train_transform = get_transforms(config['data']['image_size'], 'train')
    val_transform = get_transforms(config['data']['image_size'], 'val')
    
    # Create datasets
    train_dataset = CelebAFFHQDataset(
        root_dir=config['paths']['data_root'],
        transform=train_transform,
        mode='train',
        task='face_enhancement'
    )
    
    val_dataset = CelebAFFHQDataset(
        root_dir=config['paths']['data_root'],
        transform=val_transform,
        mode='val',
        task='face_enhancement'
    )
    
    test_dataset = CelebAFFHQDataset(
        root_dir=config['paths']['data_root'],
        transform=val_transform,
        mode='test',
        task='face_enhancement'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """Test function for dataset"""
    # Mock config
    config = {
        'data': {'image_size': 256, 'num_workers': 2, 'pin_memory': True},
        'training': {'batch_size': 4},
        'paths': {'data_root': './data'}
    }
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"Source batch shape: {batch['source'].shape}")
            print(f"Target batch shape: {batch['target'].shape}")
            break
            
    except Exception as e:
        print(f"Dataset test failed: {e}")
        print("This is expected if data directories don't exist")


if __name__ == "__main__":
    test_dataset()