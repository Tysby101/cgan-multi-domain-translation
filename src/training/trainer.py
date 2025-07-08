import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from losses.perceptual_loss import CombinedLoss, FeatureMatchingLoss
from models.discriminators.multi_scale_discriminator import MultiScaleDiscriminator
from models.generators.attention_unet import AttentionUNet
from utils.utils import compute_gradient_penalty
from utils.visualization import save_sample_images
import wandb
from tqdm import tqdm
import numpy as np
from collections import defaultdict


class CGANTrainer:
    """Trainer class for conditional GAN"""
    
    def __init__(self, config, train_loader, val_loader, device):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize models
        self._build_models()
        
        # Initialize optimizers
        self._build_optimizers()
        
        # Initialize loss functions
        self._build_losses()
        
        # Initialize training utilities
        self._build_training_utils()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_fid = float('inf')
        
    def _build_models(self):
        """Initialize generator and discriminator"""
        self.generator = AttentionUNet(
            input_channels=self.config['model']['generator']['input_channels'],
            output_channels=self.config['model']['generator']['output_channels'],
            features=self.config['model']['generator']['features'],
            attention_layers=self.config['model']['generator']['attention_layers'],
            use_spectral_norm=self.config['model']['generator']['use_spectral_norm']
        ).to(self.device)
        
        self.discriminator = MultiScaleDiscriminator(
            input_channels=self.config['model']['discriminator']['input_channels'],
            features=self.config['model']['discriminator']['features'],
            num_scales=self.config['model']['discriminator']['num_scales'],
            use_spectral_norm=self.config['model']['discriminator']['use_spectral_norm']
        ).to(self.device)
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def _build_optimizers(self):
        """Initialize optimizers"""
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config['training']['learning_rate']['generator'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['training']['learning_rate']['discriminator'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
        
        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=self.config['training']['epochs']
        )
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=self.config['training']['epochs']
        )
    
    def _build_losses(self):
        """Initialize loss functions"""
        self.criterion_combined = CombinedLoss(
            lambda_l1=self.config['training']['lambda_l1'],
            lambda_perceptual=self.config['training']['lambda_perceptual']
        ).to(self.device)
        
        self.criterion_adversarial = nn.MSELoss()
        self.criterion_feature_matching = FeatureMatchingLoss()
    
    def _build_training_utils(self):
        """Initialize training utilities"""
        # Mixed precision scaler
        if self.config.get('mixed_precision', False):
            self.scaler_g = GradScaler()
            self.scaler_d = GradScaler()
        else:
            self.scaler_g = None
            self.scaler_d = None
        
        # Create directories
        os.makedirs(self.config['paths']['checkpoints'], exist_ok=True)
        os.makedirs(self.config['paths']['results'], exist_ok=True)
        os.makedirs(self.config['paths']['logs'], exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = defaultdict(list)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Train discriminator
            d_loss, d_losses = self._train_discriminator_step(source, target)
            
            # Train generator
            g_loss, g_losses = self._train_generator_step(source, target)
            
            # Accumulate losses
            for key, value in d_losses.items():
                epoch_losses[f"d_{key}"].append(value)
            for key, value in g_losses.items():
                epoch_losses[f"g_{key}"].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f"{g_loss:.4f}",
                'D_loss': f"{d_loss:.4f}"
            })
            
            self.global_step += 1
            
            # Log to wandb
            if self.global_step % 100 == 0:
                self._log_step(d_losses, g_losses)
        
        # Calculate epoch averages
        epoch_avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return epoch_avg_losses
    
    def _train_discriminator_step(self, source, target):
        """Single discriminator training step"""
        self.optimizer_d.zero_grad()
        
        # Generate fake images
        with torch.no_grad():
            fake = self.generator(source)
        
        # Real and fake inputs for discriminator
        real_input = torch.cat([source, target], 1)
        fake_input = torch.cat([source, fake.detach()], 1)
        
        if self.scaler_d:
            with autocast():
                # Discriminator outputs
                real_outputs, real_features = self.discriminator(real_input)
                fake_outputs, fake_features = self.discriminator(fake_input)
                
                # Adversarial loss
                d_real_loss = sum([self.criterion_adversarial(output, torch.ones_like(output)) 
                                 for output in real_outputs]) / len(real_outputs)
                d_fake_loss = sum([self.criterion_adversarial(output, torch.zeros_like(output)) 
                                 for output in fake_outputs]) / len(fake_outputs)
                
                # Gradient penalty
                gp = compute_gradient_penalty(self.discriminator, real_input, fake_input, self.device)
                
                d_loss = d_real_loss + d_fake_loss + self.config['training']['lambda_gp'] * gp
            
            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
        else:
            # Standard training without mixed precision
            real_outputs, real_features = self.discriminator(real_input)
            fake_outputs, fake_features = self.discriminator(fake_input)
            
            d_real_loss = sum([self.criterion_adversarial(output, torch.ones_like(output)) 
                             for output in real_outputs]) / len(real_outputs)
            d_fake_loss = sum([self.criterion_adversarial(output, torch.zeros_like(output)) 
                             for output in fake_outputs]) / len(fake_outputs)
            
            gp = compute_gradient_penalty(self.discriminator, real_input, fake_input, self.device)
            
            d_loss = d_real_loss + d_fake_loss + self.config['training']['lambda_gp'] * gp
            
            d_loss.backward()
            self.optimizer_d.step()
        
        d_losses = {
            'total': d_loss.item(),
            'real': d_real_loss.item(),
            'fake': d_fake_loss.item(),
            'gp': gp.item()
        }
        
        return d_loss.item(), d_losses
    
    def _train_generator_step(self, source, target):
        """Single generator training step"""
        self.optimizer_g.zero_grad()
        
        if self.scaler_g:
            with autocast():
                # Generate fake images
                fake = self.generator(source)
                
                # Combined loss (L1 + Perceptual)
                combined_loss, individual_losses = self.criterion_combined(fake, target)
                
                # Adversarial loss
                fake_input = torch.cat([source, fake], 1)
                fake_outputs, fake_features = self.discriminator(fake_input)
                
                adv_loss = sum([self.criterion_adversarial(output, torch.ones_like(output)) 
                              for output in fake_outputs]) / len(fake_outputs)
                
                # Total generator loss
                g_loss = combined_loss + adv_loss
            
            self.scaler_g.scale(g_loss).backward()
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
        else:
            fake = self.generator(source)
            
            combined_loss, individual_losses = self.criterion_combined(fake, target)
            
            fake_input = torch.cat([source, fake], 1)
            fake_outputs, fake_features = self.discriminator(fake_input)
            
            adv_loss = sum([self.criterion_adversarial(output, torch.ones_like(output)) 
                          for output in fake_outputs]) / len(fake_outputs)
            
            g_loss = combined_loss + adv_loss
            
            g_loss.backward()
            self.optimizer_g.step()
        
        g_losses = individual_losses.copy()
        g_losses['adversarial'] = adv_loss.item()
        g_losses['total'] = g_loss.item()
        
        return g_loss.item(), g_losses
    
    def validate(self):
        """Validation step"""
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Generate fake images
                fake = self.generator(source)
                
                # Calculate losses
                combined_loss, individual_losses = self.criterion_combined(fake, target)
                
                for key, value in individual_losses.items():
                    val_losses[key].append(value.item())
        
        # Calculate averages
        val_avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        
        return val_avg_losses
    
    def _log_step(self, d_losses, g_losses):
        """Log step to wandb"""
        log_dict = {}
        for key, value in d_losses.items():
            log_dict[f"train/d_{key}"] = value
        for key, value in g_losses.items():
            log_dict[f"train/g_{key}"] = value
        
        log_dict["train/learning_rate_g"] = self.scheduler_g.get_last_lr()[0]
        log_dict["train/learning_rate_d"] = self.scheduler_d.get_last_lr()[0]
        
        wandb.log(log_dict, step=self.global_step)
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'best_fid': self.best_fid,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['paths']['checkpoints'], 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['paths']['checkpoints'], 'best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_fid = checkpoint['best_fid']
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            if epoch % self.config.get('validation_frequency', 5) == 0:
                val_losses = self.validate()
                
                # Log validation losses
                log_dict = {f"val/{k}": v for k, v in val_losses.items()}
                wandb.log(log_dict, step=self.global_step)
                
                # Calculate FID score
                fid_score = self._calculate_fid()
                wandb.log({"val/fid": fid_score}, step=self.global_step)
                
                # Check if best model
                is_best = fid_score < self.best_fid
                if is_best:
                    self.best_fid = fid_score
                
                # Save sample images
                self._save_samples()
                
                # Save checkpoint
                if epoch % self.config.get('save_frequency', 10) == 0:
                    self.save_checkpoint(is_best)
            
            # Update learning rates
            self.scheduler_g.step()
            self.scheduler_d.step()
    
    def _calculate_fid(self):
        """Calculate FID score on validation set"""
        # Implementation would use pytorch-fid library
        # This is a placeholder
        return np.random.random() * 50  # Placeholder
    
    def _save_samples(self):
        """Save sample generated images"""
        self.generator.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 4:  # Save only first 4 batches
                    break
                
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                fake = self.generator(source)
                
                save_sample_images(
                    source, target, fake, 
                    os.path.join(self.config['paths']['results'], 
                               f'epoch_{self.current_epoch}_batch_{i}.png')
                )


def test_trainer():
    """Test function for trainer"""
    print("Trainer module loaded successfully")
    print("Note: Full training requires data and proper configuration")


if __name__ == "__main__":
    test_trainer()