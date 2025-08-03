"""
Image diffusion model adapter for Bayesian optimization
"""

import torch
from typing import Dict, Any
import comfy.samplers
import comfy.sample

from ..core.base_optimizer import ModelAdapter


class ImageDiffusionAdapter(ModelAdapter):
    """Adapter for image diffusion models (Stable Diffusion, SDXL, Flux, etc.)"""
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return parameter space for image diffusion models"""
        return {
            # Guidance/CFG parameters
            'cfg_scale': (1.0, 20.0),
            
            # Sampling parameters
            'steps': (10, 150),
            'denoise': (0.0, 1.0),
            
            # Sampler and scheduler
            'sampler_name': [
                'euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral',
                'lms', 'dpm_fast', 'dpm_adaptive', 'dpmpp_2s_ancestral',
                'dpmpp_sde', 'dpmpp_2m', 'dpmpp_3m_sde', 'ddim', 'uni_pc'
            ],
            'scheduler': ['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform'],
            
            # Resolution parameters
            'width': (512, 2048),
            'height': (512, 2048),
            
            # Advanced parameters
            'eta': (0.0, 1.0),
            's_churn': (0.0, 100.0),
            's_tmin': (0.0, 10.0),
            's_tmax': (0.0, 999.0),
            's_noise': (0.9, 1.1),
        }
    
    def sample(self, model: Any, parameters: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Execute sampling with given parameters"""
        # Extract parameters with defaults
        cfg = parameters.get('cfg_scale', 7.0)
        steps = int(parameters.get('steps', 20))
        sampler_name = parameters.get('sampler_name', 'euler')
        scheduler = parameters.get('scheduler', 'normal')
        denoise = parameters.get('denoise', 1.0)
        seed = int(parameters.get('seed', 0))
        
        # Get conditioning and latent from kwargs
        positive = kwargs.get('positive')
        negative = kwargs.get('negative')
        latent = kwargs.get('latent')
        
        # Perform sampling using ComfyUI's common_ksampler
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
    
    def get_model_type(self) -> str:
        """Return the type of model"""
        return "image"


class FluxAdapter(ImageDiffusionAdapter):
    """Specialized adapter for Flux models with Flux-specific parameters"""
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return Flux-specific parameter space"""
        base_space = super().get_parameter_space()
        
        # Update Flux-specific ranges
        base_space.update({
            'guidance': (1.0, 10.0),  # Flux uses 'guidance' instead of 'cfg_scale'
            'steps': (20, 50),  # Flux typically needs fewer steps
            'shift': (0.0, 2.0),  # Flux-specific shift parameter
            
            # Flux-optimized samplers
            'sampler_name': ['euler', 'uni_pc', 'dpmpp_2m', 'dpmpp_3m_sde'],
            
            # Resolution ratios for Flux
            'resolution_preset': [
                '1:1', '3:4', '4:3', '9:16', '16:9',
                '2:3', '3:2', '1:2', '2:1'
            ],
        })
        
        # Remove non-Flux parameters
        base_space.pop('cfg_scale', None)
        base_space.pop('eta', None)
        base_space.pop('s_churn', None)
        base_space.pop('s_tmin', None)
        base_space.pop('s_tmax', None)
        base_space.pop('s_noise', None)
        
        return base_space
    
    def sample(self, model: Any, parameters: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Execute Flux-specific sampling"""
        # Rename guidance to cfg for compatibility
        if 'guidance' in parameters:
            parameters['cfg_scale'] = parameters.pop('guidance')
        
        # Handle resolution preset
        if 'resolution_preset' in parameters:
            preset = parameters['resolution_preset']
            width, height = self._get_resolution_from_preset(preset)
            parameters['width'] = width
            parameters['height'] = height
        
        return super().sample(model, parameters, **kwargs)
    
    def _get_resolution_from_preset(self, preset: str) -> tuple:
        """Convert resolution preset to width/height"""
        preset_map = {
            '1:1': (1024, 1024),
            '3:4': (768, 1024),
            '4:3': (1024, 768),
            '9:16': (576, 1024),
            '16:9': (1024, 576),
            '2:3': (682, 1024),
            '3:2': (1024, 682),
            '1:2': (512, 1024),
            '2:1': (1024, 512),
        }
        return preset_map.get(preset, (1024, 1024))


class SDXLAdapter(ImageDiffusionAdapter):
    """Specialized adapter for SDXL models"""
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return SDXL-specific parameter space"""
        base_space = super().get_parameter_space()
        
        # SDXL-specific parameters
        base_space.update({
            'cfg_scale': (3.0, 15.0),  # SDXL works better with lower CFG
            'steps': (20, 100),
            
            # SDXL resolution must be at least 1024
            'width': (1024, 2048),
            'height': (1024, 2048),
            
            # Refiner parameters
            'use_refiner': [True, False],
            'refiner_switch_at': (0.6, 0.9),
            'refiner_cfg': (1.0, 10.0),
            
            # SDXL conditioning
            'aesthetic_score': (1.0, 10.0),
            'negative_aesthetic_score': (1.0, 10.0),
        })
        
        return base_space