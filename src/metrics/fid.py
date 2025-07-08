from pytorch_fid.fid_score import calculate_fid_given_paths

def calculate_fid(real_path, fake_path, device='cuda', batch_size=50):
    """
    Calculate FID score between real and fake image directories
    
    Args:
        real_path: Path to directory containing real images
        fake_path: Path to directory containing fake images
        device: Device to use for computation
        batch_size: Batch size for processing
    
    Returns:
        FID score
    """
    try:
        fid_score = calculate_fid_given_paths(
            [real_path, fake_path],
            batch_size=batch_size,
            device=device,
            dims=2048
        )
        return fid_score
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return None