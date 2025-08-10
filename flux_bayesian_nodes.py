"""
Enhanced Bayesian Optimization Nodes for Flux Workflows
Specialized nodes for optimizing Flux-specific parameters
"""

import numpy as np
import torch
import json
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import matplotlib.pyplot as plt
import io

# Try to import optimization libraries
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("scikit-optimize not installed. Using basic optimization.")

class EnhancedBayesianConfig:
    """Enhanced configuration for Flux-specific parameter optimization"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "fixed_prompt": ("STRING", {"multiline": True}),
                "n_iterations": ("INT", {"default": 75, "min": 10, "max": 300}),
                "n_initial_points": ("INT", {"default": 15, "min": 5, "max": 50}),
                
                # Flux Guidance parameters
                "guidance_min": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0}),
                "guidance_max": ("FLOAT", {"default": 7.0, "min": 0.1, "max": 20.0}),
                
                # Steps parameters
                "steps_min": ("INT", {"default": 20, "min": 1, "max": 150}),
                "steps_max": ("INT", {"default": 50, "min": 1, "max": 150}),
                
                # Scheduler and sampler types
                "scheduler_types": ("STRING", {"default": "beta,normal,simple,ddim_uniform"}),
                "sampler_types": ("STRING", {"default": "euler,uni_pc,dpmpp_2m,dpmpp_3m_sde"}),
                
                # LoRA configuration
                "num_loras": ("INT", {"default": 2, "min": 0, "max": 5}),
                "lora_names": ("STRING", {"default": "", "multiline": True}),
                "lora_weight_min": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0}),
                "lora_weight_max": ("FLOAT", {"default": 1.2, "min": -2.0, "max": 2.0}),
                
                # Resolution ratios
                "resolution_ratios": ("STRING", {"default": "1:1,3:4,4:3,16:9,9:16"}),
                
                # Similarity metric
                "similarity_metric": (["Combined", "Aesthetic", "CLIP", "LPIPS", "MSE", "SSIM"],),
                
                # Seed configuration
                "optimize_seed": ("BOOLEAN", {"default": False}),
                "seed_min": ("INT", {"default": 0}),
                "seed_max": ("INT", {"default": 1000000}),
                
                "optimization_seed": ("INT", {"default": 42}),
            }
        }
    
    RETURN_TYPES = ("ENHANCED_BAYESIAN_CONFIG",)
    FUNCTION = "create_config"
    CATEGORY = "Bayesian Optimization/Flux"
    
    def create_config(self, target_image, fixed_prompt, n_iterations, n_initial_points,
                     guidance_min, guidance_max, steps_min, steps_max,
                     scheduler_types, sampler_types, num_loras, lora_names,
                     lora_weight_min, lora_weight_max, resolution_ratios,
                     similarity_metric, optimize_seed, seed_min, seed_max,
                     optimization_seed):
        
        # Parse string inputs
        schedulers = [s.strip() for s in scheduler_types.split(',') if s.strip()]
        samplers = [s.strip() for s in sampler_types.split(',') if s.strip()]
        ratios = [r.strip() for r in resolution_ratios.split(',') if r.strip()]
        lora_list = [l.strip() for l in lora_names.split('\n') if l.strip()] if lora_names else []
        
        # Create parameter space
        if SKOPT_AVAILABLE:
            space = [
                Real(guidance_min, guidance_max, name='guidance'),
                Integer(steps_min, steps_max, name='steps'),
                Categorical(schedulers, name='scheduler'),
                Categorical(samplers, name='sampler'),
                Categorical(ratios, name='resolution_ratio')
            ]
            
            # Add LoRA weights to space
            for i in range(num_loras):
                space.append(Real(lora_weight_min, lora_weight_max, name=f'lora{i+1}_weight'))
            
            # Add seed if optimizing
            if optimize_seed:
                space.append(Integer(seed_min, seed_max, name='seed'))
        else:
            space = {
                'guidance': (guidance_min, guidance_max),
                'steps': (steps_min, steps_max),
                'scheduler': schedulers,
                'sampler': samplers,
                'resolution_ratio': ratios,
            }
            
            # Add LoRA weights
            for i in range(num_loras):
                space[f'lora{i+1}_weight'] = (lora_weight_min, lora_weight_max)
            
            if optimize_seed:
                space['seed'] = (seed_min, seed_max)
        
        config = {
            "target_image": target_image,
            "fixed_prompt": fixed_prompt,
            "n_iterations": n_iterations,
            "n_initial_points": n_initial_points,
            "space": space,
            "num_loras": num_loras,
            "lora_names": lora_list,
            "similarity_metric": similarity_metric,
            "optimize_seed": optimize_seed,
            "optimization_seed": optimization_seed,
            "history": [],
            "best_params": None,
            "best_score": float('-inf'),
            "iteration": 0,
            "param_names": {
                "schedulers": schedulers,
                "samplers": samplers,
                "ratios": ratios
            }
        }
        
        return (config,)

class EnhancedParameterSampler:
    """Enhanced parameter sampler for Flux workflows"""
    
    # Class-level storage for persistent state across runs
    _optimization_state = {}
    _current_optimization_id = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
                "similarity_score": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "is_first_run": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previous_image": ("IMAGE",),  # For scoring feedback
            }
        }
    
    RETURN_TYPES = ("FLOAT", "INT", "STRING", "STRING", "STRING", 
                   "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT",
                   "INT", "ENHANCED_BAYESIAN_CONFIG", "BOOLEAN")
    RETURN_NAMES = ("guidance", "steps", "scheduler", "sampler", "resolution_ratio",
                   "lora1_weight", "lora2_weight", "lora3_weight", "lora4_weight", "lora5_weight",
                   "seed", "updated_config", "optimization_complete")
    FUNCTION = "sample_parameters"
    CATEGORY = "Bayesian Optimization/Flux"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always return a different value to force re-execution
        import time
        import random
        return f"{time.time()}_{random.random()}"
    
    def __init__(self):
        self.current_params = None
    
    def sample_parameters(self, config, similarity_score, is_first_run, previous_image=None):
        # Generate a unique ID for this optimization based on config
        import hashlib
        import os
        import pickle
        
        config_str = f"{config.get('fixed_prompt', '')}_{config.get('n_iterations', 0)}_{config.get('optimization_seed', 0)}"
        opt_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create a state file path
        try:
            import folder_paths
            temp_dir = folder_paths.get_temp_directory()
        except:
            temp_dir = "/tmp"
        
        state_file = os.path.join(temp_dir, f"bayesian_opt_{opt_id}.pkl")
        
        print(f"\n=== EnhancedParameterSampler ===")
        print(f"Optimization ID: {opt_id}")
        print(f"is_first_run: {is_first_run}")
        print(f"similarity_score: {similarity_score}")
        print(f"Config iteration: {config.get('iteration', 0)}")
        print(f"Config history length: {len(config.get('history', []))}")
        
        # Try to load existing state from file
        state = None
        if os.path.exists(state_file) and not is_first_run:
            try:
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                print(f"Loaded existing state from {state_file}")
                print(f"Loaded state: iteration {state['iteration']}, history length {len(state['history'])}")
            except Exception as e:
                print(f"Error loading state: {e}")
                state = None
        
        # Initialize new state if needed
        if state is None or is_first_run:
            if is_first_run and os.path.exists(state_file):
                os.remove(state_file)
                print(f"Removed old state file for fresh start")
            
            print(f"Initializing new optimization with ID: {opt_id}")
            state = {
                "history": [],
                "iteration": 0,
                "best_params": None,
                "best_score": float('-inf'),
                "current_params": None,
                "run_count": 0
            }
            # Reset config history for new optimization
            config["history"] = []
            config["iteration"] = 0
            config["best_params"] = None
            config["best_score"] = float('-inf')
        
        # Store state in class variable too (for within-session persistence)
        EnhancedParameterSampler._optimization_state[opt_id] = state
        
        # Sync config with persistent state
        config["history"] = state["history"]
        config["iteration"] = state["iteration"]
        config["best_params"] = state["best_params"]
        config["best_score"] = state["best_score"]
        
        # Increment run count
        state["run_count"] = state.get("run_count", 0) + 1
        
        print(f"Optimization {opt_id}: iteration {config['iteration']}, history length: {len(config['history'])}, run #{state['run_count']}")
        
        # Update history if not the actual first run and we have previous params
        if state["run_count"] > 1 and state["current_params"] is not None:
            config["history"].append({
                "params": state["current_params"],
                "score": similarity_score,
                "iteration": config["iteration"]
            })
            state["history"] = config["history"]
            
            # Update best parameters
            if similarity_score > config["best_score"]:
                config["best_score"] = similarity_score
                config["best_params"] = state["current_params"].copy()
                state["best_score"] = config["best_score"]
                state["best_params"] = config["best_params"]
                print(f"New best score: {similarity_score:.4f}")
        
        # Check if optimization is complete
        if config["iteration"] >= config["n_iterations"]:
            if config["best_params"] is not None:
                print(f"Optimization complete! Best score: {config['best_score']:.4f}")
                return self._return_params(config["best_params"], config, True)
            else:
                # Return defaults
                default_params = self._get_default_params(config)
                return self._return_params(default_params, config, True)
        
        # Sample new parameters
        if SKOPT_AVAILABLE and len(config["history"]) >= config["n_initial_points"]:
            params = self._sample_skopt_enhanced(config)
        else:
            params = self._sample_enhanced(config)
        
        # Store current params in both instance and class state
        self.current_params = params
        state["current_params"] = params
        
        # Update iteration counter
        config["iteration"] += 1
        state["iteration"] = config["iteration"]
        
        print(f"Sampled params for iteration {config['iteration']}: guidance={params.get('guidance', 'N/A'):.2f}, steps={params.get('steps', 'N/A')}")
        
        # Save state to file for persistence across runs
        try:
            with open(state_file, 'wb') as f:
                # Create a clean state dict without unpickleable objects
                save_state = {
                    "history": state["history"],
                    "iteration": state["iteration"],
                    "best_params": state["best_params"],
                    "best_score": state["best_score"],
                    "current_params": state["current_params"],
                    "run_count": state["run_count"]
                }
                pickle.dump(save_state, f)
                print(f"Saved state to {state_file}")
        except Exception as e:
            print(f"Error saving state: {e}")
        
        return self._return_params(params, config, False)
    
    def _sample_skopt_enhanced(self, config):
        """Enhanced sampling using scikit-optimize"""
        from skopt.learning import GaussianProcessRegressor
        from skopt.acquisition import gaussian_ei
        
        # Prepare data for GP
        X = []
        y = []
        
        for entry in config["history"]:
            x_point = []
            params = entry["params"]
            
            # Add continuous parameters
            x_point.append(params["guidance"])
            x_point.append(params["steps"])
            
            # Encode categorical parameters
            x_point.append(config["param_names"]["schedulers"].index(params["scheduler"]))
            x_point.append(config["param_names"]["samplers"].index(params["sampler"]))
            x_point.append(config["param_names"]["ratios"].index(params["resolution_ratio"]))
            
            # Add LoRA weights
            for i in range(config["num_loras"]):
                x_point.append(params.get(f"lora{i+1}_weight", 0.5))
            
            # Add seed if optimizing
            if config["optimize_seed"]:
                x_point.append(params.get("seed", 0))
            
            X.append(x_point)
            y.append(-entry["score"])  # Minimize negative score
        
        # Always create a fresh GP model (don't rely on instance state between runs)
        gp_model = GaussianProcessRegressor(
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        gp_model.fit(X, y)
        
        # Generate candidate points
        n_candidates = 1000
        candidates = []
        
        for _ in range(n_candidates):
            candidate = []
            
            # Sample continuous parameters
            candidate.append(np.random.uniform(
                config["space"][0].low, config["space"][0].high))  # guidance
            candidate.append(np.random.randint(
                config["space"][1].low, config["space"][1].high + 1))  # steps
            
            # Sample categorical parameters
            candidate.append(np.random.randint(0, len(config["param_names"]["schedulers"])))
            candidate.append(np.random.randint(0, len(config["param_names"]["samplers"])))
            candidate.append(np.random.randint(0, len(config["param_names"]["ratios"])))
            
            # Sample LoRA weights
            for i in range(config["num_loras"]):
                idx = 5 + i
                candidate.append(np.random.uniform(
                    config["space"][idx].low, config["space"][idx].high))
            
            # Sample seed if needed
            if config["optimize_seed"]:
                seed_idx = 5 + config["num_loras"]
                candidate.append(np.random.randint(
                    config["space"][seed_idx].low, config["space"][seed_idx].high + 1))
            
            candidates.append(candidate)
        
        # Evaluate acquisition function
        candidates = np.array(candidates)
        acquisition_values = gaussian_ei(candidates, gp_model, np.min(y))
        
        # Select best candidate
        best_idx = np.argmax(acquisition_values)
        best_candidate = candidates[best_idx]
        
        # Convert back to parameter dictionary
        params = {
            "guidance": float(best_candidate[0]),
            "steps": int(best_candidate[1]),
            "scheduler": config["param_names"]["schedulers"][int(best_candidate[2])],
            "sampler": config["param_names"]["samplers"][int(best_candidate[3])],
            "resolution_ratio": config["param_names"]["ratios"][int(best_candidate[4])],
        }
        
        # Add LoRA weights
        for i in range(config["num_loras"]):
            params[f"lora{i+1}_weight"] = float(best_candidate[5 + i])
        
        # Add seed if optimizing
        if config["optimize_seed"]:
            params["seed"] = int(best_candidate[5 + config["num_loras"]])
        else:
            params["seed"] = config["optimization_seed"]
        
        return params
    
    def _sample_enhanced(self, config):
        """Enhanced random/quasi-random sampling"""
        np.random.seed(config["optimization_seed"] + config["iteration"])
        
        if config["iteration"] < config["n_initial_points"]:
            # Use Sobol-like sequence for better coverage
            t = config["iteration"] / max(1, config["n_initial_points"] - 1)
            
            # Handle different space formats (list vs dict)
            if isinstance(config["space"], list):
                # scikit-optimize format (list of Dimension objects)
                guidance_min, guidance_max = config["space"][0].low, config["space"][0].high
                steps_min, steps_max = config["space"][1].low, config["space"][1].high
            else:
                # Dictionary format with tuples
                guidance_min, guidance_max = config["space"]["guidance"]
                steps_min, steps_max = config["space"]["steps"]
            
            # Continuous parameters
            guidance = guidance_min + t * (guidance_max - guidance_min)
            steps = int(steps_min + t * (steps_max - steps_min))
            
            # Categorical parameters - cycle through options
            scheduler_idx = config["iteration"] % len(config["param_names"]["schedulers"])
            sampler_idx = config["iteration"] % len(config["param_names"]["samplers"])
            ratio_idx = config["iteration"] % len(config["param_names"]["ratios"])
            
            params = {
                "guidance": guidance,
                "steps": steps,
                "scheduler": config["param_names"]["schedulers"][scheduler_idx],
                "sampler": config["param_names"]["samplers"][sampler_idx],
                "resolution_ratio": config["param_names"]["ratios"][ratio_idx],
            }
            
            # LoRA weights - use different patterns
            for i in range(config["num_loras"]):
                if isinstance(config["space"], list):
                    # Find the LoRA weight dimension in the list
                    lora_idx = 5 + i  # After guidance, steps, scheduler, sampler, ratio
                    if lora_idx < len(config["space"]):
                        weight_min = config["space"][lora_idx].low
                        weight_max = config["space"][lora_idx].high
                    else:
                        weight_min, weight_max = 0.0, 1.0
                else:
                    # Dictionary format
                    weight_range = config["space"].get(f"lora{i+1}_weight", (0.0, 1.0))
                    weight_min, weight_max = weight_range
                # Use sinusoidal pattern for diversity
                phase = (config["iteration"] + i * 7) / config["n_initial_points"]
                weight = weight_min + (weight_max - weight_min) * \
                        (0.5 + 0.5 * np.sin(2 * np.pi * phase))
                params[f"lora{i+1}_weight"] = float(weight)
        else:
            # Informed sampling around best point
            if config["best_params"] is not None:
                # Add noise to best params
                variance_factor = 1 - config["iteration"] / config["n_iterations"]
                
                # Get space bounds
                if isinstance(config["space"], list):
                    guidance_min, guidance_max = config["space"][0].low, config["space"][0].high
                    steps_min, steps_max = config["space"][1].low, config["space"][1].high
                else:
                    guidance_min, guidance_max = config["space"]["guidance"]
                    steps_min, steps_max = config["space"]["steps"]
                
                params = {
                    "guidance": np.clip(
                        config["best_params"]["guidance"] + 
                        np.random.normal(0, 1.0 * variance_factor),
                        guidance_min, guidance_max
                    ),
                    "steps": np.clip(
                        int(config["best_params"]["steps"] + 
                            np.random.normal(0, 5 * variance_factor)),
                        steps_min, steps_max
                    )
                }
                
                # Occasionally try different categorical values
                if np.random.random() < 0.2 * variance_factor:
                    params["scheduler"] = np.random.choice(config["param_names"]["schedulers"])
                else:
                    params["scheduler"] = config["best_params"]["scheduler"]
                
                if np.random.random() < 0.2 * variance_factor:
                    params["sampler"] = np.random.choice(config["param_names"]["samplers"])
                else:
                    params["sampler"] = config["best_params"]["sampler"]
                
                if np.random.random() < 0.1 * variance_factor:
                    params["resolution_ratio"] = np.random.choice(config["param_names"]["ratios"])
                else:
                    params["resolution_ratio"] = config["best_params"]["resolution_ratio"]
                
                # LoRA weights
                for i in range(config["num_loras"]):
                    if isinstance(config["space"], list):
                        lora_idx = 5 + i
                        if lora_idx < len(config["space"]):
                            weight_min = config["space"][lora_idx].low
                            weight_max = config["space"][lora_idx].high
                        else:
                            weight_min, weight_max = 0.0, 1.0
                    else:
                        weight_range = config["space"].get(f"lora{i+1}_weight", (0.0, 1.0))
                        weight_min, weight_max = weight_range
                    
                    params[f"lora{i+1}_weight"] = np.clip(
                        config["best_params"].get(f"lora{i+1}_weight", 0.5) +
                        np.random.normal(0, 0.2 * variance_factor),
                        weight_min, weight_max
                    )
            else:
                # Random sampling
                params = self._get_random_params(config)
        
        # Handle seed
        if config["optimize_seed"]:
            if config["iteration"] < config["n_initial_points"]:
                if isinstance(config["space"], list):
                    # Find seed dimension in list
                    seed_idx = 5 + config["num_loras"]
                    if seed_idx < len(config["space"]):
                        seed_min = config["space"][seed_idx].low
                        seed_max = config["space"][seed_idx].high
                    else:
                        seed_min, seed_max = 0, 1000000
                else:
                    seed_range = config["space"].get("seed", (0, 1000000))
                    seed_min, seed_max = seed_range
                params["seed"] = np.random.randint(seed_min, seed_max + 1)
            else:
                params["seed"] = config["optimization_seed"] + config["iteration"]
        else:
            params["seed"] = config["optimization_seed"]
        
        return params
    
    def _get_random_params(self, config):
        """Get random parameters"""
        # Get space bounds
        if isinstance(config["space"], list):
            guidance_min, guidance_max = config["space"][0].low, config["space"][0].high
            steps_min, steps_max = config["space"][1].low, config["space"][1].high
        else:
            guidance_min, guidance_max = config["space"]["guidance"]
            steps_min, steps_max = config["space"]["steps"]
        
        params = {
            "guidance": np.random.uniform(guidance_min, guidance_max),
            "steps": np.random.randint(steps_min, steps_max + 1),
            "scheduler": np.random.choice(config["param_names"]["schedulers"]),
            "sampler": np.random.choice(config["param_names"]["samplers"]),
            "resolution_ratio": np.random.choice(config["param_names"]["ratios"]),
        }
        
        for i in range(config["num_loras"]):
            if isinstance(config["space"], list):
                lora_idx = 5 + i
                if lora_idx < len(config["space"]):
                    weight_min = config["space"][lora_idx].low
                    weight_max = config["space"][lora_idx].high
                else:
                    weight_min, weight_max = 0.0, 1.0
            else:
                weight_range = config["space"].get(f"lora{i+1}_weight", (0.0, 1.0))
                weight_min, weight_max = weight_range
            params[f"lora{i+1}_weight"] = np.random.uniform(weight_min, weight_max)
        
        return params
    
    def _get_default_params(self, config):
        """Get default parameters"""
        params = {
            "guidance": 3.5,
            "steps": 30,
            "scheduler": config["param_names"]["schedulers"][0],
            "sampler": config["param_names"]["samplers"][0],
            "resolution_ratio": "1:1",
            "seed": config["optimization_seed"]
        }
        
        for i in range(config["num_loras"]):
            params[f"lora{i+1}_weight"] = 0.8
        
        return params
    
    def _return_params(self, params, config, complete):
        """Format parameters for return"""
        # Get LoRA weights, defaulting to 0 for unused slots
        lora_weights = []
        for i in range(5):  # Always return 5 LoRA weights
            if i < config["num_loras"]:
                lora_weights.append(float(params.get(f"lora{i+1}_weight", 0.0)))
            else:
                lora_weights.append(0.0)
        
        return (
            float(params["guidance"]),
            int(params["steps"]),
            params["scheduler"],
            params["sampler"],
            params["resolution_ratio"],
            lora_weights[0],
            lora_weights[1],
            lora_weights[2],
            lora_weights[3],
            lora_weights[4],
            int(params.get("seed", config["optimization_seed"])),
            config,
            complete
        )

class AestheticScorer:
    """Advanced scorer with aesthetic quality metrics"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
            },
            "optional": {
                "aesthetic_weight": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "sharpness_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "color_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("similarity_score",)
    FUNCTION = "calculate_score"
    CATEGORY = "Bayesian Optimization/Flux"
    
    def calculate_score(self, generated_image, config, 
                       aesthetic_weight=0.3, sharpness_weight=0.2, color_weight=0.2):
        target_image = config["target_image"]
        metric = config["similarity_metric"]
        
        # Convert to tensors
        if not isinstance(generated_image, torch.Tensor):
            generated_image = torch.from_numpy(generated_image).float()
        if not isinstance(target_image, torch.Tensor):
            target_image = torch.from_numpy(target_image).float()
        
        if metric == "Aesthetic":
            # Aesthetic quality scoring
            similarity = self._calculate_similarity(generated_image, target_image)
            aesthetic = self._calculate_aesthetic_score(generated_image)
            sharpness = self._calculate_sharpness(generated_image)
            color_harmony = self._calculate_color_harmony(generated_image)
            
            score = (1 - aesthetic_weight - sharpness_weight - color_weight) * similarity + \
                   aesthetic_weight * aesthetic + \
                   sharpness_weight * sharpness + \
                   color_weight * color_harmony
        else:
            # Use metrics from metrics_universal
            from .metrics_universal import UniversalMetrics
            metrics = UniversalMetrics()
            score = metrics.compute_similarity(generated_image, target_image, metric)
        
        return (float(score),)
    
    def _calculate_similarity(self, img1, img2):
        """Basic similarity calculation"""
        mse = torch.mean((img1 - img2) ** 2)
        return 1.0 / (1.0 + float(mse))
    
    def _calculate_aesthetic_score(self, image):
        """Calculate aesthetic quality score"""
        # Simple aesthetic scoring based on image statistics
        # In practice, you might use a pre-trained aesthetic model
        
        # Check color distribution
        mean_color = torch.mean(image, dim=(0, 1))
        color_variance = torch.var(image, dim=(0, 1))
        
        # Prefer balanced colors
        color_balance = 1.0 - torch.std(mean_color) * 2
        color_balance = torch.clamp(color_balance, 0, 1)
        
        # Prefer moderate variance (not too flat, not too noisy)
        optimal_variance = 0.15
        variance_score = 1.0 - torch.abs(torch.mean(color_variance) - optimal_variance) * 3
        variance_score = torch.clamp(variance_score, 0, 1)
        
        return float((color_balance + variance_score) / 2)
    
    def _calculate_sharpness(self, image):
        """Calculate image sharpness score"""
        # Convert to grayscale
        gray = torch.mean(image, dim=2)
        
        # Calculate Laplacian
        laplacian = torch.nn.functional.conv2d(
            gray.unsqueeze(0).unsqueeze(0),
            torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).float().unsqueeze(0).unsqueeze(0),
            padding=1
        )
        
        # Variance of Laplacian as sharpness measure
        sharpness = torch.var(laplacian)
        
        # Normalize to 0-1 range
        normalized_sharpness = torch.tanh(sharpness * 100)
        
        return float(normalized_sharpness)
    
    def _calculate_color_harmony(self, image):
        """Calculate color harmony score"""
        # Simple color harmony based on color wheel relationships
        
        # Convert to HSV-like representation
        max_rgb, _ = torch.max(image, dim=2)
        min_rgb, _ = torch.min(image, dim=2)
        
        # Calculate dominant hue regions
        hue_variance = torch.var(max_rgb - min_rgb)
        
        # Lower variance suggests more harmonious colors
        harmony_score = 1.0 / (1.0 + float(hue_variance) * 10)
        
        return harmony_score

class OptimizationDashboard:
    """Real-time optimization dashboard with enhanced visualizations"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
            },
            "optional": {
                "show_3d_plot": ("BOOLEAN", {"default": False}),
                "show_heatmap": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_dashboard"
    CATEGORY = "Bayesian Optimization/Flux"
    
    def create_dashboard(self, config, show_3d_plot=False, show_heatmap=True):
        if not config["history"]:
            # Return placeholder
            placeholder = np.ones((512, 512, 3), dtype=np.uint8) * 255
            return (torch.from_numpy(placeholder).float() / 255.0,)
        
        # Create figure
        n_subplots = 4 + (1 if show_3d_plot else 0) + (1 if show_heatmap else 0)
        fig = plt.figure(figsize=(16, 4 * ((n_subplots + 1) // 2)))
        
        # Main title
        fig.suptitle(
            f'Flux Parameter Optimization - Iteration {config["iteration"]}/{config["n_iterations"]}',
            fontsize=16, fontweight='bold'
        )
        
        # Extract data
        history = config["history"]
        iterations = [h["iteration"] for h in history]
        scores = [h["score"] for h in history]
        
        # Plot 1: Score progression with smoothing
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(iterations, scores, 'b-', alpha=0.5, label='Raw scores')
        
        # Add smoothed line if enough points
        if len(scores) > 5:
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(scores, sigma=2)
            ax1.plot(iterations, smoothed, 'r-', linewidth=2, label='Smoothed')
        
        # Mark best score
        if config["best_params"]:
            best_idx = scores.index(config["best_score"])
            ax1.plot(iterations[best_idx], config["best_score"], 
                    'g*', markersize=15, label=f'Best: {config["best_score"]:.4f}')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Score')
        ax1.set_title('Optimization Progress')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Parameter importance (if enough data)
        ax2 = plt.subplot(2, 3, 2)
        if len(history) > 10:
            # Calculate parameter correlations with score
            param_names = ['guidance', 'steps', 'lora1_weight']
            correlations = []
            
            for param in param_names:
                if param in history[0]["params"]:
                    values = [h["params"][param] for h in history]
                    if isinstance(values[0], (int, float)):
                        corr = np.corrcoef(values, scores)[0, 1]
                        correlations.append(corr)
                    else:
                        correlations.append(0)
                else:
                    correlations.append(0)
            
            ax2.bar(param_names, correlations)
            ax2.set_ylabel('Correlation with Score')
            ax2.set_title('Parameter Importance')
            ax2.grid(True, axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Need more data\nfor analysis', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Parameter Importance')
        
        # Plot 3: Best parameters summary
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')
        
        if config["best_params"]:
            param_text = "Best Parameters:\n\n"
            param_text += f"Score: {config['best_score']:.4f}\n"
            param_text += f"Guidance: {config['best_params']['guidance']:.2f}\n"
            param_text += f"Steps: {config['best_params']['steps']}\n"
            param_text += f"Scheduler: {config['best_params']['scheduler']}\n"
            param_text += f"Sampler: {config['best_params']['sampler']}\n"
            param_text += f"Resolution: {config['best_params']['resolution_ratio']}\n"
            
            for i in range(config["num_loras"]):
                if f"lora{i+1}_weight" in config["best_params"]:
                    param_text += f"LoRA {i+1}: {config['best_params'][f'lora{i+1}_weight']:.3f}\n"
        else:
            param_text = "No best parameters yet"
        
        ax3.text(0.1, 0.9, param_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 4: Convergence rate
        ax4 = plt.subplot(2, 3, 4)
        if len(scores) > 1:
            improvements = [0]
            current_best = scores[0]
            for score in scores[1:]:
                if score > current_best:
                    improvements.append(score - current_best)
                    current_best = score
                else:
                    improvements.append(0)
            
            ax4.plot(iterations, improvements, 'g-', marker='o')
            ax4.fill_between(iterations, 0, improvements, alpha=0.3, color='green')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Score Improvement')
            ax4.set_title('Convergence Rate')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Parameter evolution heatmap
        if show_heatmap and len(history) > 5:
            ax5 = plt.subplot(2, 3, 5)
            
            # Create parameter matrix
            param_matrix = []
            param_labels = []
            
            # Add continuous parameters
            if 'guidance' in history[0]["params"]:
                param_matrix.append([h["params"]["guidance"] for h in history])
                param_labels.append('Guidance')
            
            if 'steps' in history[0]["params"]:
                steps_normalized = [(h["params"]["steps"] - config["space"]["steps"][0]) / 
                                  (config["space"]["steps"][1] - config["space"]["steps"][0])
                                  for h in history]
                param_matrix.append(steps_normalized)
                param_labels.append('Steps (norm)')
            
            # Add LoRA weights
            for i in range(min(3, config["num_loras"])):  # Show max 3 LoRAs
                if f"lora{i+1}_weight" in history[0]["params"]:
                    param_matrix.append([h["params"][f"lora{i+1}_weight"] for h in history])
                    param_labels.append(f'LoRA {i+1}')
            
            if param_matrix:
                im = ax5.imshow(param_matrix, aspect='auto', cmap='viridis')
                ax5.set_yticks(range(len(param_labels)))
                ax5.set_yticklabels(param_labels)
                ax5.set_xlabel('Iteration')
                ax5.set_title('Parameter Evolution Heatmap')
                plt.colorbar(im, ax=ax5)
        
        # Plot 6: Exploration vs Exploitation
        ax6 = plt.subplot(2, 3, 6)
        if len(history) > config["n_initial_points"]:
            exploration_phase = iterations[:config["n_initial_points"]]
            exploitation_phase = iterations[config["n_initial_points"]:]
            
            exploration_scores = scores[:config["n_initial_points"]]
            exploitation_scores = scores[config["n_initial_points"]:]
            
            ax6.scatter(exploration_phase, exploration_scores, 
                       c='blue', label='Exploration', s=50, alpha=0.6)
            ax6.scatter(exploitation_phase, exploitation_scores, 
                       c='red', label='Exploitation', s=50, alpha=0.6)
            
            ax6.axvline(x=config["n_initial_points"], color='gray', 
                       linestyle='--', label='Phase transition')
            
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('Score')
            ax6.set_title('Exploration vs Exploitation')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close()
        
        return (torch.from_numpy(img_array).float() / 255.0,)


class IterationCounter:
    """Simple iteration counter for optimization loops"""
    
    CATEGORY = "Bayesian/Control"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_value": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "increment": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
            "optional": {
                "reset": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count"
    
    def __init__(self):
        self.counter = 0
    
    def count(self, start_value, increment, reset=False):
        """Increment counter"""
        if reset:
            self.counter = start_value
        else:
            self.counter += increment
        return (self.counter,)


class ConditionalBranch:
    """Conditional branching for workflow control"""
    
    CATEGORY = "Bayesian/Control"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN",),
            },
            "optional": {
                "true_value": ("*",),
                "false_value": ("*",),
            }
        }
    
    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("true_output", "false_output")
    FUNCTION = "branch"
    
    def branch(self, condition, true_value=None, false_value=None):
        """Branch based on condition"""
        if condition:
            return (true_value, None)
        else:
            return (None, false_value)


class BayesianResultsExporter:
    """Exports optimization results to various formats"""
    
    CATEGORY = "Bayesian/Results"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
                "filename_prefix": ("STRING", {
                    "default": "bayesian_results",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_summary",)
    FUNCTION = "export_results"
    OUTPUT_NODE = True
    
    def export_results(self, config, filename_prefix="bayesian_results"):
        """Export optimization results to JSON and generate summary"""
        import datetime
        
        # Debug print
        print(f"BayesianResultsExporter called with prefix: {filename_prefix}")
        print(f"Config keys: {config.keys() if isinstance(config, dict) else 'Not a dict'}")
        
        # Extract history from config
        history = config.get("history", [])
        
        print(f"History length: {len(history)}")
        
        if not history:
            print("Warning: No optimization history to export")
            return ("No optimization history to export - make sure to run multiple iterations with is_first_run=false after the initial run",)
        
        # Find best iteration
        best_idx = np.argmax([h["score"] for h in history])
        best_result = history[best_idx]
        
        # Helper function to convert numpy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_native(item) for item in obj)
            else:
                return obj
        
        # Convert best_result params to native types
        best_params_native = convert_to_native(best_result["params"])
        
        # Convert history to native types
        history_native = []
        for h in history:
            history_native.append({
                "params": convert_to_native(h["params"]),
                "score": float(h["score"]),
                "iteration": int(h.get("iteration", 0))
            })
        
        # Create results dictionary with native types
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_iterations": int(len(history)),
            "best_iteration": int(best_idx + 1),
            "best_score": float(best_result["score"]),
            "best_parameters": best_params_native,
            "optimization_config": {
                "n_iterations": int(config.get("n_iterations", 0)),
                "metric": str(config.get("similarity_metric", "unknown")),
                "optimize_seed": bool(config.get("optimize_seed", False)),
            },
            "history": history_native,
            "convergence_info": {
                "final_score": float(history[-1]["score"]),
                "score_improvement": float(history[-1]["score"] - history[0]["score"]),
                "converged": len(history) >= config.get("n_iterations", 0)
            }
        }
        
        # Save to JSON file in ComfyUI output directory
        import os
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{filename_prefix}_{timestamp}.json"
        
        # Try to get ComfyUI output directory, with fallback
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except (ImportError, AttributeError) as e:
            # Fallback to common paths
            if os.path.exists("/workspace/ComfyUI/output"):
                output_dir = "/workspace/ComfyUI/output"
            elif os.path.exists("output"):
                output_dir = "output"
            else:
                output_dir = "."
            print(f"Warning: Could not import folder_paths, using fallback: {output_dir}")
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, json_filename)
        
        try:
            # Use a custom JSON encoder to handle any remaining numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, np.float32)):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)
            
            with open(full_path, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            print(f"Results saved to: {full_path}")
            
            # Create summary text
            summary = f"""
Optimization Results Exported
============================
File: {json_filename}
Full Path: {full_path}
Timestamp: {results['timestamp']}

Best Result (Iteration {int(best_idx + 1)}/{len(history)}):
  Score: {float(best_result['score']):.4f}
  Guidance: {best_params_native.get('guidance', 'N/A')}
  Steps: {best_params_native.get('steps', 'N/A')}
  Scheduler: {best_params_native.get('scheduler', 'N/A')}
  Sampler: {best_params_native.get('sampler', 'N/A')}

Optimization Summary:
  Total Iterations: {len(history)}
  Initial Score: {float(history[0]['score']):.4f}
  Final Score: {float(history[-1]['score']):.4f}
  Improvement: {float(history[-1]['score'] - history[0]['score']):.4f}
  
Results saved to: {full_path}
"""
            return (summary,)
            
        except Exception as e:
            return (f"Error exporting results: {str(e)}",)


class ImageSimilarityCalculator:
    """Calculate similarity between generated and target images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "metric": (["SSIM", "MSE", "PSNR", "Combined"],),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("similarity_score", "metric_details")
    FUNCTION = "calculate_similarity"
    CATEGORY = "Bayesian Optimization"
    
    def calculate_similarity(self, generated_image, target_image, metric):
        import torch
        import numpy as np
        try:
            from skimage.metrics import structural_similarity as ssim
            from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
            SKIMAGE_AVAILABLE = True
        except ImportError:
            SKIMAGE_AVAILABLE = False
            print("scikit-image not installed. Using basic similarity metrics.")
        
        # Convert tensors to numpy arrays
        if torch.is_tensor(generated_image):
            gen_np = generated_image.cpu().numpy()
        else:
            gen_np = np.array(generated_image)
            
        if torch.is_tensor(target_image):
            target_np = target_image.cpu().numpy()
        else:
            target_np = np.array(target_image)
        
        # Handle batch dimension
        if gen_np.ndim == 4:
            gen_np = gen_np[0]
        if target_np.ndim == 4:
            target_np = target_np[0]
            
        # Ensure same shape
        if gen_np.shape != target_np.shape:
            # Resize to match target
            from PIL import Image
            if gen_np.shape[-1] == 3 or gen_np.shape[-1] == 4:
                # Channel last format
                h, w = target_np.shape[:2]
                gen_pil = Image.fromarray((gen_np * 255).astype(np.uint8))
                gen_pil = gen_pil.resize((w, h), Image.Resampling.LANCZOS)
                gen_np = np.array(gen_pil) / 255.0
            
        score = 0.0
        details = []
        
        try:
            if not SKIMAGE_AVAILABLE:
                # Fallback to basic MSE calculation
                mse = np.mean((target_np - gen_np) ** 2)
                score = 1.0 / (1.0 + mse)  # Convert to similarity score
                details.append(f"Basic MSE Score: {score:.4f}")
                details.append("Install scikit-image for more metrics")
            elif metric == "SSIM":
                # Calculate SSIM
                score = ssim(target_np, gen_np, channel_axis=-1, data_range=1.0)
                details.append(f"SSIM: {score:.4f}")
                
            elif metric == "MSE":
                # Calculate MSE (inverted to be higher = better)
                mse = mean_squared_error(target_np, gen_np)
                score = 1.0 / (1.0 + mse)  # Convert to similarity score
                details.append(f"MSE: {mse:.4f}, Score: {score:.4f}")
                
            elif metric == "PSNR":
                # Calculate PSNR (normalized to 0-1)
                psnr = peak_signal_noise_ratio(target_np, gen_np, data_range=1.0)
                score = min(psnr / 100.0, 1.0)  # Normalize to 0-1
                details.append(f"PSNR: {psnr:.2f}dB, Score: {score:.4f}")
                
            elif metric == "Combined":
                # Combine multiple metrics
                ssim_score = ssim(target_np, gen_np, channel_axis=-1, data_range=1.0)
                mse = mean_squared_error(target_np, gen_np)
                mse_score = 1.0 / (1.0 + mse)
                psnr = peak_signal_noise_ratio(target_np, gen_np, data_range=1.0)
                psnr_score = min(psnr / 100.0, 1.0)
                
                # Weighted average
                score = (ssim_score * 0.5 + mse_score * 0.25 + psnr_score * 0.25)
                details.append(f"Combined Score: {score:.4f}")
                details.append(f"  SSIM: {ssim_score:.4f}")
                details.append(f"  MSE: {mse_score:.4f}")
                details.append(f"  PSNR: {psnr_score:.4f}")
                
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            score = 0.0
            details.append(f"Error: {str(e)}")
        
        details_str = "\n".join(details)
        print(f"Similarity calculation: {details_str}")
        
        return (float(score), details_str)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "EnhancedBayesianConfig": EnhancedBayesianConfig,
    "EnhancedParameterSampler": EnhancedParameterSampler,
    "AestheticScorer": AestheticScorer,
    "OptimizationDashboard": OptimizationDashboard,
    "IterationCounter": IterationCounter,
    "ConditionalBranch": ConditionalBranch,
    "BayesianResultsExporter": BayesianResultsExporter,
    "ImageSimilarityCalculator": ImageSimilarityCalculator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedBayesianConfig": "Enhanced Bayesian Config (Flux)",
    "EnhancedParameterSampler": "Enhanced Parameter Sampler (Flux)",
    "AestheticScorer": "Aesthetic Scorer",
    "OptimizationDashboard": "Optimization Dashboard",
    "IterationCounter": "Iteration Counter",
    "ConditionalBranch": "Conditional Branch",
    "BayesianResultsExporter": "Bayesian Results Exporter",
    "ImageSimilarityCalculator": "Image Similarity Calculator",
}