"""
Adapter nodes for integrating Bayesian optimization with existing Flux workflows
These nodes convert between Bayesian optimizer outputs and Flux node inputs
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class PowerLoraAdapter:
    """Adapts Bayesian LoRA weights to Power Lora Loader format"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora1_weight": ("FLOAT", {"default": 0.0}),
                "lora2_weight": ("FLOAT", {"default": 0.0}),
                "lora3_weight": ("FLOAT", {"default": 0.0}),
                "lora4_weight": ("FLOAT", {"default": 0.0}),
                "lora5_weight": ("FLOAT", {"default": 0.0}),
                "lora_names": ("STRING", {"multiline": True}),
            },
            "optional": {
                "base_model": ("MODEL",),
                "base_clip": ("CLIP",),
            }
        }
    
    RETURN_TYPES = ("POWER_LORA_WEIGHTS", "STRING")
    RETURN_NAMES = ("lora_weights", "weight_summary")
    FUNCTION = "adapt_weights"
    CATEGORY = "Bayesian Optimization/Adapters"
    
    def adapt_weights(self, lora1_weight, lora2_weight, lora3_weight, 
                     lora4_weight, lora5_weight, lora_names,
                     base_model=None, base_clip=None):
        
        # Parse LoRA names
        lora_list = [l.strip() for l in lora_names.split('\n') if l.strip()]
        
        # Create weight dictionary for Power Lora Loader
        lora_weights = {}
        weight_summary_parts = []
        
        weights = [lora1_weight, lora2_weight, lora3_weight, lora4_weight, lora5_weight]
        
        for i, (lora_name, weight) in enumerate(zip(lora_list[:5], weights[:len(lora_list)])):
            if weight != 0.0:  # Only include non-zero weights
                lora_weights[f"lora_{i+1}"] = {
                    "name": lora_name,
                    "weight": float(weight),
                    "enabled": True
                }
                weight_summary_parts.append(f"{lora_name}: {weight:.3f}")
        
        # Create summary string
        weight_summary = "LoRA Weights: " + ", ".join(weight_summary_parts) if weight_summary_parts else "No active LoRAs"
        
        # If base model/clip provided, prepare for chaining
        if base_model is not None or base_clip is not None:
            lora_weights["_base_model"] = base_model
            lora_weights["_base_clip"] = base_clip
        
        return (lora_weights, weight_summary)

class ResolutionAdapter:
    """Converts resolution ratio string to width/height for FluxResolutionNode"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution_ratio": ("STRING", {"default": "1:1"}),
                "base_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048}),
            },
            "optional": {
                "force_multiple": ("INT", {"default": 64, "min": 8, "max": 128}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "resolution_info")
    FUNCTION = "convert_ratio"
    CATEGORY = "Bayesian Optimization/Adapters"
    
    def convert_ratio(self, resolution_ratio, base_resolution, force_multiple=64):
        # Parse ratio
        try:
            if ':' in resolution_ratio:
                w_ratio, h_ratio = resolution_ratio.split(':')
                w_ratio = float(w_ratio)
                h_ratio = float(h_ratio)
            else:
                w_ratio = h_ratio = 1.0
        except:
            w_ratio = h_ratio = 1.0
        
        # Calculate dimensions maintaining aspect ratio
        aspect = w_ratio / h_ratio
        
        if aspect >= 1:
            # Landscape or square
            width = base_resolution
            height = int(base_resolution / aspect)
        else:
            # Portrait
            height = base_resolution
            width = int(base_resolution * aspect)
        
        # Round to multiple
        width = (width // force_multiple) * force_multiple
        height = (height // force_multiple) * force_multiple
        
        # Ensure minimum size
        width = max(width, force_multiple * 8)
        height = max(height, force_multiple * 8)
        
        resolution_info = f"{width}x{height} ({resolution_ratio})"
        
        return (width, height, resolution_info)

class SchedulerAdapter:
    """Adapts scheduler name string to scheduler configuration"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler_name": ("STRING", {"default": "beta"}),
                "steps": ("INT", {"default": 30}),
            },
            "optional": {
                "beta_start": ("FLOAT", {"default": 0.00085, "min": 0.0001, "max": 0.01}),
                "beta_end": ("FLOAT", {"default": 0.012, "min": 0.001, "max": 0.1}),
            }
        }
    
    RETURN_TYPES = ("SCHEDULER_CONFIG", "STRING")
    RETURN_NAMES = ("scheduler", "scheduler_info")
    FUNCTION = "create_scheduler"
    CATEGORY = "Bayesian Optimization/Adapters"
    
    def create_scheduler(self, scheduler_name, steps, beta_start=0.00085, beta_end=0.012):
        # Map scheduler names to configurations
        scheduler_configs = {
            "beta": {
                "type": "beta",
                "beta_schedule": "scaled_linear",
                "beta_start": beta_start,
                "beta_end": beta_end,
            },
            "normal": {
                "type": "normal",
                "sigma_max": 14.0,
                "sigma_min": 0.02,
            },
            "simple": {
                "type": "simple",
                "sigma_max": 10.0,
                "sigma_min": 0.1,
            },
            "ddim_uniform": {
                "type": "ddim_uniform",
                "steps": steps,
            },
            "karras": {
                "type": "karras",
                "sigma_max": 14.0,
                "sigma_min": 0.02,
                "rho": 7.0,
            },
            "exponential": {
                "type": "exponential",
                "sigma_max": 14.0,
                "sigma_min": 0.02,
            }
        }
        
        # Get scheduler config
        if scheduler_name in scheduler_configs:
            scheduler_config = scheduler_configs[scheduler_name].copy()
        else:
            # Default to beta if unknown
            scheduler_config = scheduler_configs["beta"].copy()
            scheduler_name = "beta"
        
        # Add steps to config
        scheduler_config["steps"] = steps
        
        scheduler_info = f"{scheduler_name} scheduler, {steps} steps"
        
        return (scheduler_config, scheduler_info)

class SamplerAdapter:
    """Adapts sampler name to KSampler configuration"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": ("STRING", {"default": "euler"}),
            },
            "optional": {
                "cfg_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "sampler_options": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("SAMPLER_CONFIG", "STRING")
    RETURN_NAMES = ("sampler", "sampler_info")
    FUNCTION = "create_sampler"
    CATEGORY = "Bayesian Optimization/Adapters"
    
    def create_sampler(self, sampler_name, cfg_rescale=0.0, sampler_options=""):
        # Map sampler names to configurations
        sampler_configs = {
            "euler": {
                "name": "euler",
                "type": "standard",
                "cfg_rescale": cfg_rescale,
            },
            "euler_a": {
                "name": "euler_a",
                "type": "ancestral",
                "cfg_rescale": cfg_rescale,
            },
            "heun": {
                "name": "heun",
                "type": "standard",
                "cfg_rescale": cfg_rescale,
            },
            "dpm_2": {
                "name": "dpm_2",
                "type": "standard",
                "cfg_rescale": cfg_rescale,
            },
            "dpm_2_a": {
                "name": "dpm_2_a",
                "type": "ancestral",
                "cfg_rescale": cfg_rescale,
            },
            "dpmpp_2s_a": {
                "name": "dpmpp_2s_a",
                "type": "ancestral",
                "cfg_rescale": cfg_rescale,
            },
            "dpmpp_2m": {
                "name": "dpmpp_2m",
                "type": "standard",
                "cfg_rescale": cfg_rescale,
            },
            "dpmpp_sde": {
                "name": "dpmpp_sde",
                "type": "sde",
                "cfg_rescale": cfg_rescale,
            },
            "dpmpp_3m_sde": {
                "name": "dpmpp_3m_sde",
                "type": "sde",
                "cfg_rescale": cfg_rescale,
                "sde_options": {
                    "s_noise": 1.0,
                    "eta": 1.0,
                }
            },
            "uni_pc": {
                "name": "uni_pc",
                "type": "predictor_corrector",
                "cfg_rescale": cfg_rescale,
                "order": 2,
            },
            "uni_pc_bh2": {
                "name": "uni_pc_bh2",
                "type": "predictor_corrector",
                "cfg_rescale": cfg_rescale,
                "order": 2,
                "bh2": True,
            }
        }
        
        # Get sampler config
        if sampler_name in sampler_configs:
            sampler_config = sampler_configs[sampler_name].copy()
        else:
            # Default to euler if unknown
            sampler_config = sampler_configs["euler"].copy()
            sampler_name = "euler"
        
        # Parse additional options if provided
        if sampler_options:
            try:
                import json
                additional_options = json.loads(sampler_options)
                sampler_config.update(additional_options)
            except:
                pass
        
        sampler_info = f"{sampler_name} sampler"
        if sampler_config.get("type") == "sde":
            sampler_info += " (SDE)"
        elif sampler_config.get("type") == "ancestral":
            sampler_info += " (Ancestral)"
        
        return (sampler_config, sampler_info)

class OptimizationLoopController:
    """Controls the optimization loop flow"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "optimization_complete": ("BOOLEAN",),
                "current_iteration": ("INT", {"default": 0}),
                "max_iterations": ("INT", {"default": 50}),
            },
            "optional": {
                "early_stop_threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
                "current_score": ("FLOAT", {"default": 0.0}),
                "patience": ("INT", {"default": 10, "min": 1, "max": 50}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("continue_optimization", "save_checkpoint", "status_message")
    FUNCTION = "control_loop"
    CATEGORY = "Bayesian Optimization/Adapters"
    
    def __init__(self):
        self.best_score = 0.0
        self.iterations_without_improvement = 0
    
    def control_loop(self, optimization_complete, current_iteration, max_iterations,
                    early_stop_threshold=0.95, current_score=0.0, patience=10):
        
        # Check if optimization is complete
        if optimization_complete:
            return (False, True, f"Optimization complete! Best score: {self.best_score:.4f}")
        
        # Update best score tracking
        if current_score > self.best_score:
            self.best_score = current_score
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        
        # Check early stopping conditions
        if self.best_score >= early_stop_threshold:
            return (False, True, f"Early stop: Reached threshold {early_stop_threshold:.2f}")
        
        if self.iterations_without_improvement >= patience:
            return (False, True, f"Early stop: No improvement for {patience} iterations")
        
        if current_iteration >= max_iterations:
            return (False, True, f"Reached maximum iterations ({max_iterations})")
        
        # Continue optimization
        save_checkpoint = (current_iteration % 10 == 0)  # Save every 10 iterations
        
        status = f"Iteration {current_iteration}/{max_iterations} | "
        status += f"Current: {current_score:.4f} | Best: {self.best_score:.4f}"
        
        return (True, save_checkpoint, status)

class ParameterLogger:
    """Logs optimization parameters and results"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iteration": ("INT",),
                "score": ("FLOAT",),
                "guidance": ("FLOAT",),
                "steps": ("INT",),
                "scheduler": ("STRING",),
                "sampler": ("STRING",),
            },
            "optional": {
                "log_file": ("STRING", {"default": "optimization_log.csv"}),
                "append": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log_entry",)
    FUNCTION = "log_parameters"
    CATEGORY = "Bayesian Optimization/Adapters"
    
    def log_parameters(self, iteration, score, guidance, steps, scheduler, sampler,
                      log_file="optimization_log.csv", append=True):
        
        import csv
        import os
        from datetime import datetime
        
        # Create log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "iteration": iteration,
            "score": score,
            "guidance": guidance,
            "steps": steps,
            "scheduler": scheduler,
            "sampler": sampler
        }
        
        # Write to CSV file
        file_exists = os.path.exists(log_file)
        mode = 'a' if append and file_exists else 'w'
        
        with open(log_file, mode, newline='') as csvfile:
            fieldnames = ["timestamp", "iteration", "score", "guidance", 
                         "steps", "scheduler", "sampler"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists or not append:
                writer.writeheader()
            
            writer.writerow(log_entry)
        
        # Create formatted string for display
        log_string = f"[{timestamp}] Iter {iteration}: Score={score:.4f}, "
        log_string += f"Guidance={guidance:.2f}, Steps={steps}, "
        log_string += f"Scheduler={scheduler}, Sampler={sampler}"
        
        return (log_string,)

class BatchParameterGenerator:
    """Generates multiple parameter sets for parallel evaluation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 16}),
                "generation_strategy": (["sobol", "random", "grid", "latin_hypercube"],),
            }
        }
    
    RETURN_TYPES = ("PARAMETER_BATCH",)
    FUNCTION = "generate_batch"
    CATEGORY = "Bayesian Optimization/Adapters"
    
    def generate_batch(self, config, batch_size, generation_strategy):
        
        batch = []
        
        if generation_strategy == "sobol":
            # Generate Sobol sequence points
            from scipy.stats import qmc
            
            # Create bounds for continuous parameters
            n_dims = 2 + config["num_loras"]  # guidance, steps, loras
            if config["optimize_seed"]:
                n_dims += 1
            
            sampler = qmc.Sobol(d=n_dims, scramble=True)
            samples = sampler.random(n=batch_size)
            
            for i in range(batch_size):
                params = {
                    "guidance": float(samples[i, 0] * (config["space"]["guidance"][1] - 
                                                     config["space"]["guidance"][0]) + 
                                    config["space"]["guidance"][0]),
                    "steps": int(samples[i, 1] * (config["space"]["steps"][1] - 
                                                config["space"]["steps"][0]) + 
                               config["space"]["steps"][0]),
                    "scheduler": np.random.choice(config["param_names"]["schedulers"]),
                    "sampler": np.random.choice(config["param_names"]["samplers"]),
                    "resolution_ratio": np.random.choice(config["param_names"]["ratios"]),
                }
                
                # Add LoRA weights
                for j in range(config["num_loras"]):
                    weight_range = config["space"][f"lora{j+1}_weight"]
                    params[f"lora{j+1}_weight"] = float(
                        samples[i, 2 + j] * (weight_range[1] - weight_range[0]) + weight_range[0]
                    )
                
                batch.append(params)
                
        elif generation_strategy == "random":
            # Pure random sampling
            for _ in range(batch_size):
                params = {
                    "guidance": np.random.uniform(
                        config["space"]["guidance"][0],
                        config["space"]["guidance"][1]
                    ),
                    "steps": np.random.randint(
                        config["space"]["steps"][0],
                        config["space"]["steps"][1] + 1
                    ),
                    "scheduler": np.random.choice(config["param_names"]["schedulers"]),
                    "sampler": np.random.choice(config["param_names"]["samplers"]),
                    "resolution_ratio": np.random.choice(config["param_names"]["ratios"]),
                }
                
                for j in range(config["num_loras"]):
                    weight_range = config["space"][f"lora{j+1}_weight"]
                    params[f"lora{j+1}_weight"] = np.random.uniform(weight_range[0], weight_range[1])
                
                batch.append(params)
                
        elif generation_strategy == "grid":
            # Simple grid sampling
            # Create a simplified grid for batch_size points
            n_per_dim = int(np.ceil(batch_size ** (1/3)))  # Approximate cube root
            
            guidance_values = np.linspace(
                config["space"]["guidance"][0],
                config["space"]["guidance"][1],
                n_per_dim
            )
            steps_values = np.linspace(
                config["space"]["steps"][0],
                config["space"]["steps"][1],
                n_per_dim
            ).astype(int)
            
            count = 0
            for g in guidance_values:
                for s in steps_values:
                    if count >= batch_size:
                        break
                    
                    params = {
                        "guidance": float(g),
                        "steps": int(s),
                        "scheduler": config["param_names"]["schedulers"][
                            count % len(config["param_names"]["schedulers"])
                        ],
                        "sampler": config["param_names"]["samplers"][
                            count % len(config["param_names"]["samplers"])
                        ],
                        "resolution_ratio": config["param_names"]["ratios"][
                            count % len(config["param_names"]["ratios"])
                        ],
                    }
                    
                    # Add LoRA weights
                    for j in range(config["num_loras"]):
                        weight_range = config["space"][f"lora{j+1}_weight"]
                        # Use a simple pattern for grid
                        weight = weight_range[0] + (weight_range[1] - weight_range[0]) * \
                                (count % 3) / 2
                        params[f"lora{j+1}_weight"] = float(weight)
                    
                    batch.append(params)
                    count += 1
                    
        elif generation_strategy == "latin_hypercube":
            # Latin Hypercube Sampling
            from scipy.stats import qmc
            
            n_dims = 2 + config["num_loras"]
            sampler = qmc.LatinHypercube(d=n_dims)
            samples = sampler.random(n=batch_size)
            
            for i in range(batch_size):
                params = {
                    "guidance": float(samples[i, 0] * (config["space"]["guidance"][1] - 
                                                     config["space"]["guidance"][0]) + 
                                    config["space"]["guidance"][0]),
                    "steps": int(samples[i, 1] * (config["space"]["steps"][1] - 
                                                config["space"]["steps"][0]) + 
                               config["space"]["steps"][0]),
                    "scheduler": config["param_names"]["schedulers"][
                        int(samples[i, 0] * len(config["param_names"]["schedulers"]))
                    ],
                    "sampler": config["param_names"]["samplers"][
                        int(samples[i, 1] * len(config["param_names"]["samplers"]))
                    ],
                    "resolution_ratio": np.random.choice(config["param_names"]["ratios"]),
                }
                
                # Add LoRA weights
                for j in range(config["num_loras"]):
                    weight_range = config["space"][f"lora{j+1}_weight"]
                    params[f"lora{j+1}_weight"] = float(
                        samples[i, 2 + j] * (weight_range[1] - weight_range[0]) + weight_range[0]
                    )
                
                batch.append(params)
        
        return ({"batch": batch, "size": batch_size, "strategy": generation_strategy},)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "PowerLoraAdapter": PowerLoraAdapter,
    "ResolutionAdapter": ResolutionAdapter,
    "SchedulerAdapter": SchedulerAdapter,
    "SamplerAdapter": SamplerAdapter,
    "OptimizationLoopController": OptimizationLoopController,
    "ParameterLogger": ParameterLogger,
    "BatchParameterGenerator": BatchParameterGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerLoraAdapter": "Power LoRA Adapter",
    "ResolutionAdapter": "Resolution Adapter",
    "SchedulerAdapter": "Scheduler Adapter", 
    "SamplerAdapter": "Sampler Adapter",
    "OptimizationLoopController": "Optimization Loop Controller",
    "ParameterLogger": "Parameter Logger",
    "BatchParameterGenerator": "Batch Parameter Generator",
}