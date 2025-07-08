import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lpips


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self, layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'], 
                 weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Layer mapping
        self.layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35
        }
        
        self.target_layers = [self.layer_map[layer] for layer in layers]
        self.weights = weights
        
        # Normalization for ImageNet pre-trained models
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def extract_features(self, x):
        """Extract features from specified layers"""
        # Normalize input
        x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        x = (x - self.mean) / self.std
        
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.target_layers:
                features.append(x)
        
        return features
    
    def forward(self, pred, target):
        """Compute perceptual loss"""
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            loss += self.weights[i] * F.mse_loss(pred_feat, target_feat)
        
        return loss


class LPIPSLoss(nn.Module):
    """LPIPS (Learned Perceptual Image Patch Similarity) loss"""
    def __init__(self, net='alex'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)
        
        # Freeze parameters
        for param in self.lpips.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        """Compute LPIPS loss"""
        return self.lpips(pred, target).mean()


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for discriminator features"""
    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
    
    def forward(self, real_features, fake_features):
        """
        Compute feature matching loss
        
        Args:
            real_features: List of feature maps from real images
            fake_features: List of feature maps from fake images
        """
        loss = 0
        
        for scale in range(self.num_scales):
            for i in range(len(real_features[scale])):
                real_feat = real_features[scale][i]
                fake_feat = fake_features[scale][i]
                loss += F.l1_loss(fake_feat, real_feat.detach())
        
        return loss


class StyleLoss(nn.Module):
    """Style loss using Gram matrices"""
    def __init__(self):
        super().__init__()
    
    def gram_matrix(self, x):
        """Compute Gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred, target):
        """Compute style loss"""
        pred_gram = self.gram_matrix(pred)
        target_gram = self.gram_matrix(target)
        return F.mse_loss(pred_gram, target_gram)


class ContentLoss(nn.Module):
    """Content loss using high-level features"""
    def __init__(self):
        super().__init__()
        # Use pre-trained ResNet features
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        """Compute content loss"""
        # Normalize inputs
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        
        pred_features = self.features(pred)
        target_features = self.features(target)
        
        return F.mse_loss(pred_features, target_features)


class CombinedLoss(nn.Module):
    """Combined loss function for cGAN training"""
    def __init__(self, lambda_l1=100.0, lambda_perceptual=10.0, 
                 lambda_style=1.0, lambda_content=1.0):
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.style_loss = StyleLoss()
        self.content_loss = ContentLoss()
    
    def forward(self, pred, target):
        """Compute combined loss"""
        losses = {}
        
        # L1 loss
        l1 = self.l1_loss(pred, target)
        losses['l1'] = l1
        
        # Perceptual loss
        perceptual = self.perceptual_loss(pred, target)
        losses['perceptual'] = perceptual
        
        # Style loss
        style = self.style_loss(pred, target)
        losses['style'] = style
        
        # Content loss
        content = self.content_loss(pred, target)
        losses['content'] = content
        
        # Total loss
        total = (self.lambda_l1 * l1 + 
                self.lambda_perceptual * perceptual + 
                self.lambda_style * style + 
                self.lambda_content * content)
        
        losses['total'] = total
        
        return total, losses


def test_losses():
    """Test function for loss modules"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test data
    pred = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    
    # Test VGG perceptual loss
    vgg_loss = VGGPerceptualLoss().to(device)
    vgg_result = vgg_loss(pred, target)
    print(f"VGG Perceptual Loss: {vgg_result.item():.4f}")
    
    # Test LPIPS loss
    lpips_loss = LPIPSLoss().to(device)
    lpips_result = lpips_loss(pred, target)
    print(f"LPIPS Loss: {lpips_result.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedLoss().to(device)
    total_loss, individual_losses = combined_loss(pred, target)
    print(f"Combined Loss: {total_loss.item():.4f}")
    for name, loss in individual_losses.items():
        print(f"  {name}: {loss.item():.4f}")


if __name__ == "__main__":
    test_losses()