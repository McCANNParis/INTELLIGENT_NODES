import torch
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from .bayesian_optimizer import BayesianOptimizer
from .metrics import ImageMetrics

# ComfyUI imports
import comfy.samplers
import comfy.sample


class ParameterSpaceNode:
    CATEGORY = "Bayesian/Optimization"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg_min": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "cfg_max": ("FLOAT", {"default": 15.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "steps_min": ("INT", {"default": 10, "min": 1, "max": 150}),
                "steps_max": ("INT", {"default": 50, "min": 1, "max": 150}),
                "denoise_min": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "denoise_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "samplers": ("STRING", {"default": "euler,dpmpp_2m,ddim", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("PARAM_SPACE",)
    RETURN_NAMES = ("parameter_space",)
    FUNCTION = "define_space"
    
    def define_space(self, cfg_min, cfg_max, steps_min, steps_max, denoise_min, denoise_max, samplers):
        param_space = {
            'cfg_scale': (cfg_min, cfg_max),
            'steps': (steps_min, steps_max),
            'denoise': (denoise_min, denoise_max),
            'sampler_name': [s.strip() for s in samplers.split(',')]
        }
        
        return (param_space,)


class BayesianOptimizerNode:
    CATEGORY = "Bayesian/Optimization"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parameter_space": ("PARAM_SPACE",),
                "optimization_steps": ("INT", {"default": 20, "min": 5, "max": 100}),
                "acquisition_function": (["EI", "UCB", "PI"], {"default": "EI"}),
                "exploration_weight": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "previous_history": ("BAYES_HISTORY",),
            }
        }
    
    RETURN_TYPES = ("BAYES_OPTIMIZER", "BAYES_HISTORY")
    RETURN_NAMES = ("optimizer", "history")
    FUNCTION = "create_optimizer"
    
    def create_optimizer(self, parameter_space, optimization_steps, acquisition_function, 
                        exploration_weight, previous_history=None):
        # Create optimizer
        optimizer = BayesianOptimizer(parameter_space)
        
        # Load previous history if provided
        if previous_history is not None:
            optimizer.load_history(previous_history)
        
        # Store configuration
        config = {
            "optimization_steps": optimization_steps,
            "acquisition_function": acquisition_function,
            "exploration_weight": exploration_weight,
            "current_step": len(optimizer.X_observed)
        }
        
        # Create optimizer object with config
        optimizer_obj = {
            "optimizer": optimizer,
            "config": config
        }
        
        return (optimizer_obj, optimizer.get_history())


class MetricEvaluatorNode:
    CATEGORY = "Bayesian/Optimization"
    
    def __init__(self):
        self.metrics_calculator = ImageMetrics()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "metrics": ("STRING", {"default": "SSIM,Aesthetic", "multiline": False}),
            },
            "optional": {
                "target_image": ("IMAGE",),
                "metric_weights": ("STRING", {"default": "{}", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("score", "metrics_details")
    FUNCTION = "evaluate"
    
    def evaluate(self, generated_image, metrics, target_image=None, metric_weights="{}"):
        # Parse metrics list
        metric_list = [m.strip() for m in metrics.split(',')]
        
        # Parse weights
        try:
            weights = json.loads(metric_weights) if metric_weights else {}
        except:
            weights = {}
        
        # Compute combined score
        score = self.metrics_calculator.compute_combined_score(
            generated_image,
            target_image,
            metric_list,
            weights
        )
        
        # Create detailed metrics report
        details = {
            "combined_score": score,
            "metrics_used": metric_list,
            "weights": weights
        }
        
        if target_image is not None:
            individual_scores = {}
            for metric in metric_list:
                if metric == "SSIM":
                    individual_scores[metric] = self.metrics_calculator.compute_ssim(generated_image, target_image)
                elif metric == "LPIPS":
                    individual_scores[metric] = self.metrics_calculator.compute_lpips(generated_image, target_image)
                elif metric == "CLIP":
                    individual_scores[metric] = self.metrics_calculator.compute_clip_similarity(generated_image, target_image)
                elif metric == "PSNR":
                    individual_scores[metric] = self.metrics_calculator.compute_psnr(generated_image, target_image)
            details["individual_scores"] = individual_scores
        
        return (score, json.dumps(details, indent=2))


class BayesianSamplerNode:
    CATEGORY = "Bayesian/Optimization"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "optimizer": ("BAYES_OPTIMIZER",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "previous_score": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "manual_override": ("STRING", {"default": "{}", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING", "BAYES_OPTIMIZER")
    RETURN_NAMES = ("samples", "parameters_used", "updated_optimizer")
    FUNCTION = "sample"
    
    def sample(self, model, positive, negative, latent_image, optimizer, seed, 
              previous_score=0.0, manual_override="{}"):
        # Extract optimizer and config
        opt = optimizer["optimizer"]
        config = optimizer["config"]
        
        # Update with previous score if this is not the first iteration
        if config["current_step"] > 0 and hasattr(self, '_last_params'):
            opt.update(self._last_params, previous_score)
        
        # Get next parameters suggestion
        if config["current_step"] < config["optimization_steps"]:
            suggested_params = opt.suggest_next(acquisition=config["acquisition_function"])
        else:
            # Use best parameters after optimization is complete
            suggested_params = opt.best_params if opt.best_params else opt.sample_random()
        
        # Apply manual overrides if provided
        try:
            overrides = json.loads(manual_override) if manual_override else {}
            suggested_params.update(overrides)
        except:
            pass
        
        # Store parameters for next iteration
        self._last_params = suggested_params.copy()
        
        # Extract parameters
        cfg = suggested_params.get('cfg_scale', 7.0)
        steps = int(suggested_params.get('steps', 20))
        sampler_name = suggested_params.get('sampler_name', 'euler')
        denoise = suggested_params.get('denoise', 1.0)
        
        # Perform sampling
        samples = comfy.sample.common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler="normal",
            positive=positive,
            negative=negative,
            latent=latent_image,
            denoise=denoise
        )[0]
        
        # Update optimizer state
        config["current_step"] += 1
        updated_optimizer = {
            "optimizer": opt,
            "config": config
        }
        
        # Create parameters report
        params_report = {
            "iteration": config["current_step"],
            "parameters": suggested_params,
            "best_score_so_far": opt.best_score,
            "best_params_so_far": opt.best_params
        }
        
        return (samples, json.dumps(params_report, indent=2), updated_optimizer)


# Additional utility nodes

class BayesHistoryLoaderNode:
    CATEGORY = "Bayesian/Optimization"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "history_json": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("BAYES_HISTORY",)
    RETURN_NAMES = ("history",)
    FUNCTION = "load_history"
    
    def load_history(self, history_json):
        try:
            history = json.loads(history_json)
        except:
            history = {
                'X_observed': [],
                'y_observed': [],
                'best_params': None,
                'best_score': -np.inf,
                'param_space': {}
            }
        
        return (history,)


class BayesHistorySaverNode:
    CATEGORY = "Bayesian/Optimization"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "optimizer": ("BAYES_OPTIMIZER",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("history_json",)
    FUNCTION = "save_history"
    OUTPUT_NODE = True
    
    def save_history(self, optimizer):
        opt = optimizer["optimizer"]
        history = opt.get_history()
        
        return (json.dumps(history, indent=2),)