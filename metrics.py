import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("LPIPS not available. Install with: pip install lpips")

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class ImageMetrics:
    def __init__(self):
        self.clip_model = None
        self.lpips_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def init_clip(self):
        """Initialize CLIP model if available"""
        if CLIP_AVAILABLE and self.clip_model is None:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
    def init_lpips(self):
        """Initialize LPIPS model if available"""
        if LPIPS_AVAILABLE and self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Remove batch dimension
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)  # CHW to HWC
        
        return tensor.cpu().numpy()
    
    def numpy_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor"""
        if len(array.shape) == 2:
            array = np.stack([array] * 3, axis=-1)  # Grayscale to RGB
        if array.shape[-1] == 3:
            array = array.transpose(2, 0, 1)  # HWC to CHW
        
        tensor = torch.from_numpy(array).float()
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def compute_clip_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute CLIP feature similarity between two images"""
        if not CLIP_AVAILABLE:
            return 0.0
            
        self.init_clip()
        if self.clip_model is None:
            return 0.0
        
        # Ensure tensors are on correct device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features1 = self.clip_model.encode_image(img1)
            features2 = self.clip_model.encode_image(img2)
            
            # Normalize features
            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(features1, features2, dim=-1)
            
        return similarity.item()
    
    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute LPIPS perceptual distance (lower is better)"""
        if not LPIPS_AVAILABLE:
            return 1.0
            
        self.init_lpips()
        if self.lpips_model is None:
            return 1.0
        
        # Ensure tensors are on correct device and normalized to [-1, 1]
        img1 = img1.to(self.device) * 2 - 1
        img2 = img2.to(self.device) * 2 - 1
        
        with torch.no_grad():
            distance = self.lpips_model(img1, img2)
            
        return distance.item()
    
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute Structural Similarity Index (higher is better)"""
        # Convert to numpy
        img1_np = self.tensor_to_numpy(img1)
        img2_np = self.tensor_to_numpy(img2)
        
        # Ensure images are in [0, 1] range
        img1_np = np.clip(img1_np, 0, 1)
        img2_np = np.clip(img2_np, 0, 1)
        
        # Compute SSIM
        return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)
    
    def compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio (higher is better)"""
        # Convert to numpy
        img1_np = self.tensor_to_numpy(img1)
        img2_np = self.tensor_to_numpy(img2)
        
        # Ensure images are in [0, 1] range
        img1_np = np.clip(img1_np, 0, 1)
        img2_np = np.clip(img2_np, 0, 1)
        
        # Compute PSNR
        return psnr(img1_np, img2_np, data_range=1.0)
    
    def compute_mse(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute Mean Squared Error (lower is better)"""
        return torch.mean((img1 - img2) ** 2).item()
    
    def compute_mae(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute Mean Absolute Error (lower is better)"""
        return torch.mean(torch.abs(img1 - img2)).item()
    
    def compute_aesthetic_score(self, img: torch.Tensor) -> float:
        """Placeholder for aesthetic score computation"""
        # This would require a specialized aesthetic scoring model
        # For now, return a dummy score based on basic image statistics
        
        img_np = self.tensor_to_numpy(img)
        
        # Simple heuristics for aesthetic score
        # Higher contrast and balanced brightness tend to be more aesthetic
        brightness = np.mean(img_np)
        contrast = np.std(img_np)
        
        # Prefer images that aren't too dark or too bright
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        
        # Prefer images with moderate contrast
        contrast_score = min(contrast * 4, 1.0)
        
        return (brightness_score + contrast_score) / 2
    
    def compute_combined_score(self, 
                            generated_image: torch.Tensor,
                            target_image: Optional[torch.Tensor] = None,
                            metrics: List[str] = None,
                            weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted combination of multiple metrics"""
        if metrics is None:
            metrics = ["SSIM", "LPIPS"]
        
        if weights is None:
            weights = {metric: 1.0 / len(metrics) for metric in metrics}
        
        scores = {}
        
        for metric in metrics:
            if metric == "CLIP" and target_image is not None:
                scores[metric] = self.compute_clip_similarity(generated_image, target_image)
            elif metric == "LPIPS" and target_image is not None:
                # LPIPS is a distance, so we invert it
                scores[metric] = 1.0 - min(self.compute_lpips(generated_image, target_image), 1.0)
            elif metric == "SSIM" and target_image is not None:
                scores[metric] = self.compute_ssim(generated_image, target_image)
            elif metric == "PSNR" and target_image is not None:
                # Normalize PSNR to [0, 1] range (assuming max PSNR of 50)
                scores[metric] = min(self.compute_psnr(generated_image, target_image) / 50.0, 1.0)
            elif metric == "Aesthetic":
                scores[metric] = self.compute_aesthetic_score(generated_image)
            elif metric == "MSE" and target_image is not None:
                # MSE is an error, so we invert it
                scores[metric] = 1.0 - min(self.compute_mse(generated_image, target_image), 1.0)
            elif metric == "MAE" and target_image is not None:
                # MAE is an error, so we invert it
                scores[metric] = 1.0 - min(self.compute_mae(generated_image, target_image), 1.0)
        
        # Compute weighted sum
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 1.0 / len(scores))
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0