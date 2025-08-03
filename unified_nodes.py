"""
Unified node system that combines all optimization capabilities
"""

import json
import torch
import numpy as np
from typing import Dict, Any, Optional

from .core import (
    UniversalParameterSpace,
    UniversalBayesianOptimizer,
    UniversalSampler,
    SimpleOptimizationSetup,
    SimpleOptimizationRun
)
from .metrics_universal import MetricEvaluatorUniversal
from .visualization_export_nodes import UniversalVisualizationNode, UniversalExportNode


# Auto-detection nodes
class AutoModelOptimizer:
    """Automatic model detection and optimization setup"""
    
    CATEGORY = "Bayesian/Auto"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "target": ("IMAGE,VIDEO",),
                "steps": ("INT", {"default": 30, "min": 10, "max": 200}),
            },
            "optional": {
                "advanced_config": ("STRING", {"default": "{}", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "DICT", "STRING")
    RETURN_NAMES = ("optimized_output", "best_parameters", "report")
    FUNCTION = "optimize"
    
    def optimize(self, model, positive, negative, latent, target, steps, 
                advanced_config="{}"):
        # Parse config
        try:
            config = json.loads(advanced_config)
        except:
            config = {}
        
        # Auto-detect model type
        model_type = self._detect_model_type(model)
        
        # Create parameter space
        space_node = UniversalParameterSpace()
        param_space, adapter = space_node.define_space(
            model_type=model_type,
            optimize_seed=config.get('optimize_seed', False),
            seed_min=config.get('seed_min', 0),
            seed_max=config.get('seed_max', 1000000),
            custom_params=json.dumps(config.get('custom_params', {})),
            model=model
        )
        
        # Create optimizer
        opt_node = UniversalBayesianOptimizer()
        optimizer, history = opt_node.create_optimizer(
            parameter_space=param_space,
            model_adapter=adapter,
            optimization_steps=steps,
            initial_samples=max(5, steps // 10),
            acquisition_function=config.get('acquisition', 'EI'),
            exploration_weight=config.get('exploration', 0.1)
        )
        
        # Create metric evaluator
        metric_eval = MetricEvaluatorUniversal()
        
        # Run optimization loop
        sampler = UniversalSampler()
        best_score = -float('inf')
        best_output = latent
        best_params = {}
        
        for i in range(steps):
            # Sample
            output, params, optimizer = sampler.sample(
                model=model,
                positive=positive,
                negative=negative,
                latent=latent,
                optimizer=optimizer,
                previous_score=best_score if i > 0 else 0.0
            )
            
            # Evaluate
            score, details = metric_eval.evaluate(
                generated_content=output['samples'],
                target_content=target,
                metric=config.get('metric', 'auto')
            )
            
            # Update best
            if score > best_score:
                best_score = score
                best_output = output
                best_params = params
        
        # Create report
        report = f"""Optimization Complete
Model Type: {model_type}
Total Steps: {steps}
Best Score: {best_score:.4f}
Best Parameters: {json.dumps(best_params, indent=2)}
"""
        
        return (best_output, best_params, report)
    
    def _detect_model_type(self, model) -> str:
        """Detect model type from model object"""
        model_name = model.__class__.__name__.lower()
        if 'video' in model_name or 'animate' in model_name:
            return 'video'
        return 'image'


# Preset configurations for common use cases
class PresetOptimizationConfigs:
    """Pre-configured optimization setups for common scenarios"""
    
    CATEGORY = "Bayesian/Presets"
    
    PRESETS = {
        "quality_focus": {
            "metric": "Aesthetic",
            "steps": 50,
            "exploration": 0.1,
            "optimize_seed": False,
            "custom_params": {
                "cfg_scale": [3.0, 12.0],
                "steps": [20, 80],
            }
        },
        "speed_focus": {
            "metric": "SSIM",
            "steps": 20,
            "exploration": 0.2,
            "optimize_seed": False,
            "custom_params": {
                "cfg_scale": [4.0, 8.0],
                "steps": [10, 30],
            }
        },
        "accuracy_focus": {
            "metric": "Combined",
            "steps": 100,
            "exploration": 0.05,
            "optimize_seed": True,
            "custom_params": {
                "cfg_scale": [1.0, 15.0],
                "steps": [20, 100],
            }
        },
        "video_quality": {
            "metric": "Temporal",
            "steps": 40,
            "exploration": 0.15,
            "optimize_seed": False,
            "custom_params": {
                "num_frames": [8, 32],
                "temporal_weight": [0.3, 0.8],
                "motion_scale": [0.5, 1.5],
            }
        },
        "flux_optimized": {
            "metric": "Combined",
            "steps": 30,
            "exploration": 0.1,
            "optimize_seed": False,
            "custom_params": {
                "guidance": [1.0, 7.0],
                "steps": [20, 50],
                "shift": [0.0, 1.5],
            }
        }
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()),),
            },
            "optional": {
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 500}),
                "override_metric": (["default", "SSIM", "LPIPS", "Aesthetic", "Combined"],),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("config_json",)
    FUNCTION = "get_config"
    
    def get_config(self, preset, override_steps=-1, override_metric="default"):
        config = self.PRESETS[preset].copy()
        
        if override_steps > 0:
            config["steps"] = override_steps
        
        if override_metric != "default":
            config["metric"] = override_metric
        
        return (json.dumps(config, indent=2),)


# Batch optimization for multiple targets
class BatchOptimizer:
    """Optimize for multiple target images/videos simultaneously"""
    
    CATEGORY = "Bayesian/Batch"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "targets": ("IMAGE,VIDEO",),  # Batch of targets
                "batch_mode": (["average", "best_of", "weighted"], {"default": "average"}),
                "steps_per_target": ("INT", {"default": 10, "min": 5, "max": 50}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "DICT", "IMAGE")
    RETURN_NAMES = ("optimized_output", "parameters", "comparison_grid")
    FUNCTION = "batch_optimize"
    
    def batch_optimize(self, model, positive, negative, latent, targets,
                      batch_mode, steps_per_target):
        # This is a placeholder - actual implementation would handle batch optimization
        return (latent, {}, torch.zeros(1, 512, 512, 3))


# Main node mappings combining all systems
NODE_CLASS_MAPPINGS = {
    # Core universal nodes
    "UniversalParameterSpace": UniversalParameterSpace,
    "UniversalBayesianOptimizer": UniversalBayesianOptimizer,
    "UniversalSampler": UniversalSampler,
    
    # Simple workflow nodes
    "SimpleOptimizationSetup": SimpleOptimizationSetup,
    "SimpleOptimizationRun": SimpleOptimizationRun,
    
    # Metric evaluation
    "MetricEvaluatorUniversal": MetricEvaluatorUniversal,
    
    # Auto and preset nodes
    "AutoModelOptimizer": AutoModelOptimizer,
    "PresetOptimizationConfigs": PresetOptimizationConfigs,
    "BatchOptimizer": BatchOptimizer,
    
    # Visualization (from existing)
    "UniversalVisualizationNode": UniversalVisualizationNode,
    "UniversalExportNode": UniversalExportNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Core
    "UniversalParameterSpace": "Universal Parameter Space",
    "UniversalBayesianOptimizer": "Universal Bayesian Optimizer",
    "UniversalSampler": "Universal Sampler",
    
    # Simple
    "SimpleOptimizationSetup": "Simple Optimization Setup",
    "SimpleOptimizationRun": "Simple Optimization Run",
    
    # Metrics
    "MetricEvaluatorUniversal": "Universal Metric Evaluator",
    
    # Auto/Presets
    "AutoModelOptimizer": "Auto Model Optimizer",
    "PresetOptimizationConfigs": "Optimization Presets",
    "BatchOptimizer": "Batch Optimizer",
    
    # Visualization
    "UniversalVisualizationNode": "Universal Visualization",
    "UniversalExportNode": "Universal Export",
}