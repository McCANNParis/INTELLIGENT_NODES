"""
Video diffusion model adapter for Bayesian optimization
"""

import torch
from typing import Dict, Any, List
import numpy as np

from ..core.base_optimizer import ModelAdapter


class VideoDiffusionAdapter(ModelAdapter):
    """Adapter for video diffusion models (AnimateDiff, ModelScope, etc.)"""
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return parameter space for video diffusion models"""
        return {
            # Video-specific parameters
            'num_frames': (8, 128),
            'fps': (8, 30),
            'motion_scale': (0.5, 2.0),
            
            # Temporal consistency
            'temporal_weight': (0.0, 1.0),
            'frame_overlap': (0, 4),
            'motion_blur': (0.0, 1.0),
            
            # Standard diffusion parameters
            'cfg_scale': (3.0, 15.0),
            'steps': (20, 100),
            'denoise': (0.8, 1.0),
            
            # Sampling
            'sampler_name': [
                'euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_sde',
                'ddim', 'uni_pc'
            ],
            'scheduler': ['normal', 'karras', 'linear', 'cosine'],
            
            # Resolution
            'width': (512, 1280),
            'height': (512, 720),
            
            # Video-specific samplers
            'temporal_sampler': ['uniform', 'pyramid', 'sliding_window'],
            'keyframe_interval': (4, 16),
            
            # Motion parameters
            'camera_motion': ['none', 'pan_left', 'pan_right', 'zoom_in', 'zoom_out', 'rotate'],
            'motion_strength': (0.0, 1.0),
        }
    
    def sample(self, model: Any, parameters: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Execute video sampling with given parameters"""
        # Extract video-specific parameters
        num_frames = int(parameters.get('num_frames', 16))
        fps = int(parameters.get('fps', 8))
        motion_scale = parameters.get('motion_scale', 1.0)
        temporal_weight = parameters.get('temporal_weight', 0.5)
        
        # Standard parameters
        cfg = parameters.get('cfg_scale', 7.0)
        steps = int(parameters.get('steps', 20))
        sampler_name = parameters.get('sampler_name', 'euler')
        scheduler = parameters.get('scheduler', 'normal')
        denoise = parameters.get('denoise', 1.0)
        seed = int(parameters.get('seed', 0))
        
        # Get inputs
        positive = kwargs.get('positive')
        negative = kwargs.get('negative')
        latent = kwargs.get('latent')
        
        # Prepare video latent if needed
        video_latent = self._prepare_video_latent(latent, num_frames)
        
        # Apply temporal conditioning
        positive_video = self._apply_temporal_conditioning(
            positive, num_frames, temporal_weight, motion_scale
        )
        
        # Sample video frames
        try:
            # Try to use video-specific sampling if available
            if hasattr(model, 'sample_video'):
                samples = model.sample_video(
                    positive=positive_video,
                    negative=negative,
                    latent=video_latent,
                    cfg=cfg,
                    steps=steps,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    denoise=denoise,
                    seed=seed,
                    num_frames=num_frames,
                    fps=fps
                )
            else:
                # Fallback to frame-by-frame sampling
                samples = self._sample_frames(
                    model=model,
                    positive=positive_video,
                    negative=negative,
                    latent=video_latent,
                    parameters=parameters,
                    num_frames=num_frames
                )
        except Exception as e:
            print(f"Video sampling failed, falling back to image mode: {e}")
            # Fallback to single frame
            import comfy.sample
            samples = comfy.sample.common_ksampler(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                denoise=denoise
            )[0]
        
        return samples
    
    def _prepare_video_latent(self, latent: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Prepare latent for video generation"""
        if len(latent['samples'].shape) == 4:  # Single frame latent
            # Repeat for video frames
            batch, channels, height, width = latent['samples'].shape
            video_latent = latent['samples'].unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
            return {'samples': video_latent}
        return latent
    
    def _apply_temporal_conditioning(self, conditioning: Any, num_frames: int, 
                                   temporal_weight: float, motion_scale: float) -> Any:
        """Apply temporal conditioning for video generation"""
        # This is a simplified version - actual implementation would depend on model
        return conditioning
    
    def _sample_frames(self, model: Any, positive: Any, negative: Any, 
                      latent: Dict, parameters: Dict[str, Any], num_frames: int) -> torch.Tensor:
        """Sample video frames individually with temporal coherence"""
        import comfy.sample
        
        frames = []
        prev_frame = None
        
        for i in range(num_frames):
            # Modify conditioning based on previous frame for coherence
            if prev_frame is not None and parameters.get('temporal_weight', 0) > 0:
                # Simple temporal coherence - actual implementation would be more sophisticated
                frame_latent = {
                    'samples': latent['samples'] * (1 - parameters['temporal_weight']) + 
                              prev_frame * parameters['temporal_weight']
                }
            else:
                frame_latent = latent
            
            # Sample frame
            frame = comfy.sample.common_ksampler(
                model=model,
                seed=int(parameters.get('seed', 0)) + i,
                steps=int(parameters.get('steps', 20)),
                cfg=parameters.get('cfg_scale', 7.0),
                sampler_name=parameters.get('sampler_name', 'euler'),
                scheduler=parameters.get('scheduler', 'normal'),
                positive=positive,
                negative=negative,
                latent=frame_latent,
                denoise=parameters.get('denoise', 1.0)
            )[0]
            
            frames.append(frame)
            prev_frame = frame['samples']
        
        # Stack frames
        return {'samples': torch.stack([f['samples'] for f in frames], dim=2)}
    
    def get_model_type(self) -> str:
        """Return the type of model"""
        return "video"


class AnimateDiffAdapter(VideoDiffusionAdapter):
    """Specialized adapter for AnimateDiff models"""
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return AnimateDiff-specific parameter space"""
        base_space = super().get_parameter_space()
        
        base_space.update({
            # AnimateDiff specific
            'context_length': (8, 32),
            'context_overlap': (2, 8),
            'motion_module_scale': (0.0, 2.0),
            
            # Motion LoRA weights
            'motion_lora_scale': (0.0, 2.0),
            
            # Interpolation
            'interpolation_factor': (1, 4),
            'smoothing_factor': (0.0, 1.0),
        })
        
        return base_space