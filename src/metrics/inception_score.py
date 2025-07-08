import torch
import numpy as np

def calculate_inception_score(images, splits=10, batch_size=32):
    """
    Calculate Inception Score for generated images
    
    Args:
        images: Tensor of images (N, C, H, W) in range [-1, 1]
        splits: Number of splits for IS calculation
        batch_size: Batch size for processing
    
    Returns:
        IS mean and standard deviation
    """
    from torchvision.models import inception_v3
    from torch.nn.functional import softmax
    
    # Load pre-trained Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model = inception_model.cuda()
    
    # Convert images to [0, 1] range
    images = (images + 1) / 2
    
    # Resize images to 299x299 for Inception
    images = torch.nn.functional.interpolate(
        images, size=(299, 299), mode='bilinear', align_corners=False
    )
    
    # Get predictions
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            pred = inception_model(batch)
            pred = softmax(pred, dim=1)
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Calculate IS
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits)]
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
        kl_div = np.mean(np.sum(kl_div, axis=1))
        scores.append(np.exp(kl_div))
    
    return np.mean(scores), np.std(scores)