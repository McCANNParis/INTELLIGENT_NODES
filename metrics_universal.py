"""
Universal metrics system that works with images, videos, and other modalities
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Create inline ImageMetrics class to replace removed metrics.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


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
        if not SKIMAGE_AVAILABLE:
            # Simple fallback SSIM
            diff = torch.mean((img1 - img2) ** 2)
            return float(1.0 / (1.0 + diff))
        
        # Convert to numpy
        img1_np = self.tensor_to_numpy(img1)
        img2_np = self.tensor_to_numpy(img2)
        
        # Ensure images are in [0, 1] range
        img1_np = np.clip(img1_np, 0, 1)
        img2_np = np.clip(img2_np, 0, 1)
        
        # Compute SSIM
        return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)
    
    def compute_aesthetic_score(self, img: torch.Tensor) -> float:
        """Placeholder for aesthetic score computation"""
        img_np = self.tensor_to_numpy(img)
        
        # Simple heuristics for aesthetic score
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
            elif metric == "Aesthetic":
                scores[metric] = self.compute_aesthetic_score(generated_image)
        
        # Compute weighted sum
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 1.0 / len(scores))
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


class UniversalMetrics:
    """Metrics calculator that adapts to different content types"""
    
    def __init__(self):
        self.image_metrics = ImageMetrics()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def detect_content_type(self, content: torch.Tensor) -> str:
        """Detect whether content is image, video, or other"""
        shape = content.shape
        
        if len(shape) == 4:  # B, C, H, W
            return "image"
        elif len(shape) == 5:  # B, C, T, H, W or B, T, C, H, W
            return "video"
        elif len(shape) == 3:  # B, C, L (could be audio)
            return "audio"
        else:
            return "unknown"
    
    def compute_similarity(self, generated: torch.Tensor, target: torch.Tensor,
                         metric: str = "auto", content_type: Optional[str] = None) -> float:
        """Compute similarity between generated and target content"""
        # Auto-detect content type if not specified
        if content_type is None:
            content_type = self.detect_content_type(generated)
        
        if content_type == "image":
            return self._compute_image_similarity(generated, target, metric)
        elif content_type == "video":
            return self._compute_video_similarity(generated, target, metric)
        elif content_type == "audio":
            return self._compute_audio_similarity(generated, target, metric)
        else:
            # Fallback to basic similarity
            return self._compute_basic_similarity(generated, target)
    
    def _compute_image_similarity(self, img1: torch.Tensor, img2: torch.Tensor, 
                                metric: str) -> float:
        """Compute image similarity using existing metrics"""
        if metric == "auto" or metric == "Combined":
            metrics_to_use = ["SSIM", "LPIPS", "Aesthetic"]
            return self.image_metrics.compute_combined_score(
                img1, img2, metrics_to_use
            )
        else:
            # Use specific metric
            return self.image_metrics.compute_combined_score(
                img1, img2, [metric]
            )
    
    def _compute_video_similarity(self, vid1: torch.Tensor, vid2: torch.Tensor,
                                metric: str) -> float:
        """Compute video similarity"""
        # Handle different video tensor formats
        if vid1.shape[2] > 3:  # B, C, T, H, W format
            vid1 = vid1.permute(0, 2, 1, 3, 4)  # -> B, T, C, H, W
            vid2 = vid2.permute(0, 2, 1, 3, 4)
        
        batch_size, num_frames = vid1.shape[:2]
        
        # Compute frame-wise similarity
        frame_scores = []
        for i in range(num_frames):
            frame1 = vid1[:, i]  # B, C, H, W
            frame2 = vid2[:, i]
            score = self._compute_image_similarity(frame1, frame2, metric)
            frame_scores.append(score)
        
        # Compute temporal consistency
        temporal_score = self._compute_temporal_consistency(vid1, vid2)
        
        # Combine frame similarity and temporal consistency
        frame_avg = np.mean(frame_scores)
        combined_score = 0.8 * frame_avg + 0.2 * temporal_score
        
        return float(combined_score)
    
    def _compute_temporal_consistency(self, vid1: torch.Tensor, vid2: torch.Tensor) -> float:
        """Compute temporal consistency between videos"""
        # Simple temporal consistency based on frame differences
        if vid1.shape[1] < 2:  # Not enough frames
            return 1.0
        
        # Compute frame-to-frame differences
        diff1 = torch.mean(torch.abs(vid1[:, 1:] - vid1[:, :-1]))
        diff2 = torch.mean(torch.abs(vid2[:, 1:] - vid2[:, :-1]))
        
        # Similar temporal dynamics = higher score
        consistency = 1.0 - torch.abs(diff1 - diff2) / (diff1 + diff2 + 1e-6)
        
        return float(consistency)
    
    def _compute_audio_similarity(self, aud1: torch.Tensor, aud2: torch.Tensor,
                                metric: str) -> float:
        """Compute audio similarity (placeholder)"""
        # Simple MSE-based similarity for now
        mse = torch.mean((aud1 - aud2) ** 2)
        return float(1.0 / (1.0 + mse))
    
    def _compute_basic_similarity(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        """Basic similarity for unknown content types"""
        # Flatten and compute cosine similarity
        t1_flat = t1.flatten()
        t2_flat = t2.flatten()
        
        similarity = torch.nn.functional.cosine_similarity(
            t1_flat.unsqueeze(0),
            t2_flat.unsqueeze(0)
        )
        
        return float(similarity)
    
    def compute_quality_score(self, content: torch.Tensor, 
                            content_type: Optional[str] = None) -> float:
        """Compute quality score without reference"""
        if content_type is None:
            content_type = self.detect_content_type(content)
        
        if content_type == "image":
            return self.image_metrics.compute_aesthetic_score(content)
        elif content_type == "video":
            # Average quality across frames
            if content.shape[2] > 3:  # B, C, T, H, W format
                content = content.permute(0, 2, 1, 3, 4)
            
            scores = []
            for i in range(content.shape[1]):
                frame_score = self.image_metrics.compute_aesthetic_score(content[:, i])
                scores.append(frame_score)
            
            return float(np.mean(scores))
        else:
            # Basic quality metric
            return self._compute_basic_quality(content)
    
    def _compute_basic_quality(self, content: torch.Tensor) -> float:
        """Basic quality metric for unknown content"""
        # Check for diversity (not all same value)
        std = torch.std(content)
        diversity_score = torch.tanh(std * 10)
        
        # Check for reasonable range
        range_score = 1.0 - torch.abs(torch.mean(content) - 0.5) * 2
        
        return float((diversity_score + range_score) / 2)


class MetricEvaluatorUniversal:
    """Universal metric evaluator node"""
    
    CATEGORY = "Bayesian/Universal"
    
    def __init__(self):
        self.metrics = UniversalMetrics()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_content": ("IMAGE,VIDEO,AUDIO,LATENT",),
                "metric": (["auto", "SSIM", "LPIPS", "Aesthetic", "Temporal", "Combined"],),
            },
            "optional": {
                "target_content": ("IMAGE,VIDEO,AUDIO,LATENT",),
                "content_type": (["auto", "image", "video", "audio"],),
                "weights": ("STRING", {"default": "{}", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "DICT")
    RETURN_NAMES = ("score", "details")
    FUNCTION = "evaluate"
    
    def evaluate(self, generated_content, metric="auto", target_content=None,
                content_type="auto", weights="{}"):
        # Detect content type
        if content_type == "auto":
            detected_type = self.metrics.detect_content_type(generated_content)
        else:
            detected_type = content_type
        
        # Parse weights
        try:
            import json
            weight_dict = json.loads(weights) if weights else {}
        except:
            weight_dict = {}
        
        # Compute score
        if target_content is not None:
            # Similarity score
            score = self.metrics.compute_similarity(
                generated_content, target_content, metric, detected_type
            )
        else:
            # Quality score
            score = self.metrics.compute_quality_score(
                generated_content, detected_type
            )
        
        # Create details
        details = {
            "score": float(score),
            "content_type": detected_type,
            "metric": metric,
            "has_target": target_content is not None
        }
        
        return (float(score), details)