"""
Base classes for Bayesian optimization that work with any diffusion model
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
# Simple inline optimizer to replace the removed bayesian_optimizer.py
class BayesianOptimizer:
    """Simple Bayesian optimizer for parameter optimization"""
    
    def __init__(self, parameter_space):
        self.parameter_space = parameter_space
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -float('inf')
    
    def suggest_next(self, acquisition="EI"):
        """Suggest next parameters to try"""
        # Simple random sampling for now
        return self.sample_random()
    
    def sample_random(self):
        """Sample random parameters from space"""
        import random
        params = {}
        for key, value in self.parameter_space.items():
            if isinstance(value, tuple) and len(value) == 2:
                # Numeric range
                if isinstance(value[0], int) and isinstance(value[1], int):
                    params[key] = random.randint(value[0], value[1])
                else:
                    params[key] = random.uniform(value[0], value[1])
            elif isinstance(value, list):
                # Categorical
                params[key] = random.choice(value)
            else:
                params[key] = value
        return params
    
    def update(self, params, score):
        """Update optimizer with new observation"""
        self.X_observed.append(params)
        self.y_observed.append(score)
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
    
    def get_history(self):
        """Get optimization history"""
        return {
            'X_observed': self.X_observed,
            'y_observed': self.y_observed,
            'best_params': self.best_params,
            'best_score': self.best_score
        }
    
    def load_history(self, history):
        """Load previous optimization history"""
        self.X_observed = history.get('X_observed', [])
        self.y_observed = history.get('y_observed', [])
        self.best_params = history.get('best_params', None)
        self.best_score = history.get('best_score', -float('inf'))


class ModelAdapter(ABC):
    """Abstract base class for adapting different diffusion models"""
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return the parameter space for optimization"""
        pass
    
    @abstractmethod
    def sample(self, model: Any, parameters: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Execute sampling with given parameters"""
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Return the type of model (image/video/audio)"""
        pass


class BaseOptimizationNode:
    """Base class for optimization nodes that work with any model type"""
    
    CATEGORY = "Bayesian/Universal"
    
    def __init__(self):
        self.model_adapter: Optional[ModelAdapter] = None
        self.optimizer: Optional[BayesianOptimizer] = None
    
    def set_model_adapter(self, adapter: ModelAdapter):
        """Set the model adapter for this node"""
        self.model_adapter = adapter
    
    def get_base_parameter_space(self) -> Dict[str, Any]:
        """Get base parameters common to all models"""
        return {
            'seed': (0, 2**32 - 1),
            'batch_size': (1, 8),
        }


class UniversalParameterSpace(BaseOptimizationNode):
    """Universal parameter space definition that adapts to model type"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["auto", "image", "video", "audio"], {"default": "auto"}),
                "optimize_seed": ("BOOLEAN", {"default": False}),
                "seed_min": ("INT", {"default": 0, "min": 0}),
                "seed_max": ("INT", {"default": 1000000, "min": 0}),
                "custom_params": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("PARAM_SPACE", "MODEL_ADAPTER")
    RETURN_NAMES = ("parameter_space", "model_adapter")
    FUNCTION = "define_space"
    
    def define_space(self, model_type: str, optimize_seed: bool, seed_min: int, 
                    seed_max: int, custom_params: str, model=None):
        # Auto-detect model type if needed
        if model_type == "auto" and model is not None:
            model_type = self._detect_model_type(model)
        
        # Create appropriate adapter
        adapter = self._create_adapter(model_type)
        
        # Get model-specific parameter space
        param_space = adapter.get_parameter_space()
        
        # Add base parameters
        base_params = self.get_base_parameter_space()
        if optimize_seed:
            base_params['seed'] = (seed_min, seed_max)
        else:
            base_params.pop('seed', None)
        
        param_space.update(base_params)
        
        # Parse and add custom parameters
        try:
            import json
            custom = json.loads(custom_params)
            for key, value in custom.items():
                if isinstance(value, list) and len(value) == 2:
                    param_space[key] = tuple(value)
                else:
                    param_space[key] = value
        except:
            pass
        
        return (param_space, adapter)
    
    def _detect_model_type(self, model) -> str:
        """Auto-detect model type from model object"""
        model_class = model.__class__.__name__.lower()
        
        if any(x in model_class for x in ['video', 'temporal', 'motion']):
            return "video"
        elif any(x in model_class for x in ['audio', 'sound', 'speech']):
            return "audio"
        else:
            return "image"
    
    def _create_adapter(self, model_type: str) -> ModelAdapter:
        """Create appropriate model adapter"""
        if model_type == "image":
            from ..adapters.image_adapter import ImageDiffusionAdapter
            return ImageDiffusionAdapter()
        elif model_type == "video":
            from ..adapters.video_adapter import VideoDiffusionAdapter
            return VideoDiffusionAdapter()
        elif model_type == "audio":
            from ..adapters.audio_adapter import AudioDiffusionAdapter
            return AudioDiffusionAdapter()
        else:
            from ..adapters.image_adapter import ImageDiffusionAdapter
            return ImageDiffusionAdapter()


class UniversalBayesianOptimizer(BaseOptimizationNode):
    """Universal Bayesian optimizer that works with any model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parameter_space": ("PARAM_SPACE",),
                "model_adapter": ("MODEL_ADAPTER",),
                "optimization_steps": ("INT", {"default": 20, "min": 5, "max": 200}),
                "initial_samples": ("INT", {"default": 5, "min": 3, "max": 50}),
                "acquisition_function": (["EI", "UCB", "PI"], {"default": "EI"}),
                "exploration_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "previous_history": ("OPTIMIZATION_HISTORY",),
            }
        }
    
    RETURN_TYPES = ("OPTIMIZER", "OPTIMIZATION_HISTORY")
    RETURN_NAMES = ("optimizer", "history")
    FUNCTION = "create_optimizer"
    
    def create_optimizer(self, parameter_space, model_adapter, optimization_steps,
                        initial_samples, acquisition_function, exploration_weight,
                        previous_history=None):
        # Create optimizer
        optimizer = BayesianOptimizer(parameter_space)
        
        # Load previous history if provided
        if previous_history is not None:
            optimizer.load_history(previous_history)
        
        # Create optimizer object with all necessary info
        optimizer_obj = {
            "optimizer": optimizer,
            "adapter": model_adapter,
            "config": {
                "optimization_steps": optimization_steps,
                "initial_samples": initial_samples,
                "acquisition_function": acquisition_function,
                "exploration_weight": exploration_weight,
                "current_step": len(optimizer.X_observed)
            }
        }
        
        return (optimizer_obj, optimizer.get_history())


class UniversalSampler(BaseOptimizationNode):
    """Universal sampler that works with any diffusion model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "optimizer": ("OPTIMIZER",),
            },
            "optional": {
                "previous_score": ("FLOAT", {"default": 0.0}),
                "manual_params": ("STRING", {"default": "{}", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "DICT", "OPTIMIZER")
    RETURN_NAMES = ("samples", "parameters", "updated_optimizer")
    FUNCTION = "sample"
    
    def __init__(self):
        self._last_params = None
    
    def sample(self, model, positive, negative, latent, optimizer,
              previous_score=0.0, manual_params="{}"):
        opt = optimizer["optimizer"]
        adapter = optimizer["adapter"]
        config = optimizer["config"]
        
        # Update with previous score if not first iteration
        if config["current_step"] > 0 and self._last_params is not None:
            opt.update(self._last_params, previous_score)
        
        # Get next parameters
        if config["current_step"] < config["initial_samples"]:
            # Random sampling for initial points
            suggested_params = opt.sample_random()
        elif config["current_step"] < config["optimization_steps"]:
            # Bayesian optimization
            suggested_params = opt.suggest_next(
                acquisition=config["acquisition_function"]
            )
        else:
            # Use best parameters after optimization
            suggested_params = opt.best_params if opt.best_params else opt.sample_random()
        
        # Apply manual overrides
        try:
            import json
            overrides = json.loads(manual_params) if manual_params else {}
            suggested_params.update(overrides)
        except:
            pass
        
        # Store parameters for next iteration
        self._last_params = suggested_params.copy()
        
        # Sample using adapter
        samples = adapter.sample(
            model=model,
            parameters=suggested_params,
            positive=positive,
            negative=negative,
            latent=latent
        )
        
        # Update optimizer state
        config["current_step"] += 1
        updated_optimizer = {
            "optimizer": opt,
            "adapter": adapter,
            "config": config
        }
        
        return (samples, suggested_params, updated_optimizer)


# Simplified workflow nodes
class SimpleOptimizationSetup:
    """Simplified setup for basic optimization workflows"""
    
    CATEGORY = "Bayesian/Simple"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "target_image": ("IMAGE",),
                "optimization_steps": ("INT", {"default": 20, "min": 5, "max": 100}),
                "metric": (["SSIM", "LPIPS", "Aesthetic", "Combined"], {"default": "Combined"}),
            }
        }
    
    RETURN_TYPES = ("SIMPLE_CONFIG",)
    FUNCTION = "setup"
    
    def setup(self, model, target_image, optimization_steps, metric):
        # Auto-detect model type and create configuration
        model_type = self._detect_model_type(model)
        
        config = {
            "model": model,
            "target_image": target_image,
            "optimization_steps": optimization_steps,
            "metric": metric,
            "model_type": model_type,
            "iteration": 0,
            "history": [],
            "best_score": -float('inf'),
            "best_params": None
        }
        
        return (config,)
    
    def _detect_model_type(self, model) -> str:
        """Simple model type detection"""
        model_name = model.__class__.__name__.lower()
        if 'video' in model_name:
            return 'video'
        return 'image'


class SimpleOptimizationRun:
    """Simple one-click optimization runner"""
    
    CATEGORY = "Bayesian/Simple"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("SIMPLE_CONFIG",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "FLOAT", "SIMPLE_CONFIG")
    RETURN_NAMES = ("optimized_latent", "best_score", "final_config")
    FUNCTION = "run_optimization"
    
    def run_optimization(self, config, positive, negative, latent):
        # This would run the full optimization loop
        # For now, return placeholder
        return (latent, 0.95, config)