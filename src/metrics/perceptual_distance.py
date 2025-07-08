

class PerceptualDistance:
    """Class for calculating perceptual distances using different networks"""
    
    def __init__(self, network='vgg', device='cuda'):
        self.device = device
        self.network = network
        
        if network == 'vgg':
            from torchvision.models import vgg19
            self.model = vgg19(pretrained=True).features[:35].eval().to(device)
            self.normalize = torch.nn.functional.normalize
        elif network == 'alex':
            self.model = lpips.LPIPS(net='alex').to(device)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def calculate_distance(self, img1, img2):
        """Calculate perceptual distance between two images"""
        if self.network == 'vgg':
            # Extract VGG features
            feat1 = self.model(img1)
            feat2 = self.model(img2)
            
            # Calculate cosine distance
            feat1_flat = feat1.view(feat1.size(0), -1)
            feat2_flat = feat2.view(feat2.size(0), -1)
            
            feat1_norm = self.normalize(feat1_flat, p=2, dim=1)
            feat2_norm = self.normalize(feat2_flat, p=2, dim=1)
            
            distance = 1 - torch.sum(feat1_norm * feat2_norm, dim=1)
            return distance.mean().item()
        
        elif self.network == 'alex':
            with torch.no_grad():
                distance = self.model(img1, img2)
            return distance.mean().item()