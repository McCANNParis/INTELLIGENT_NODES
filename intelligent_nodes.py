"""
Intelligent Self-Optimizing Nodes for ComfyUI
Transforms workflows into self-improving machines that learn with each run
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError("Please install optuna: pip install optuna")

try:
    from dreamsim import dreamsim
    DREAMSIM_AVAILABLE = True
except ImportError:
    DREAMSIM_AVAILABLE = False
    print("DreamSim not available. Install with: pip install dreamsim")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not available. Install with: pip install lpips")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available")

# Custom data type definitions for ComfyUI
class StudyWrapper:
    """Wrapper for Optuna Study object to pass between nodes"""
    def __init__(self, study):
        self.study = study
        self.study_id = id(study)

class TrialWrapper:
    """Wrapper for Optuna Trial object to pass between nodes"""
    def __init__(self, trial, study_id):
        self.trial = trial
        self.study_id = study_id
        self.trial_id = trial.number if trial else None

# Global storage for persistent studies (survives node re-creation)
PERSISTENT_STUDIES = {}

def get_studies_dir():
    """Get the directory for saving study files"""
    # Try ComfyUI models directory first
    comfy_models = Path("models/optimizers")
    if not comfy_models.exists():
        # Fallback to local directory
        comfy_models = Path("optimizers")
    comfy_models.mkdir(parents=True, exist_ok=True)
    return comfy_models

class OptimizerStateNode:
    """
    Central brain of the optimization system.
    Creates and manages the Optuna study with persistent storage.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "study_name": ("STRING", {"default": "my_optimization"}),
                "direction": (["minimize", "maximize"],),
                "sampler": (["TPE", "Random", "CmaEs"],),
                "pruner": (["None", "MedianPruner", "HyperbandPruner"],),
            },
            "optional": {
                "reset": ("BOOLEAN", {"default": False}),
                "load_best": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STUDY", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("study", "n_trials", "best_value", "status")
    FUNCTION = "manage_study"
    CATEGORY = "Optimization/Core"
    
    def manage_study(self, study_name, direction, sampler, pruner, reset=False, load_best=False):
        """Create or load a study with persistence"""
        
        # Generate unique study ID
        study_id = hashlib.md5(study_name.encode()).hexdigest()[:8]
        save_path = get_studies_dir() / f"{study_name}_{study_id}.pkl"
        
        # Reset if requested
        if reset and save_path.exists():
            save_path.unlink()
            if study_id in PERSISTENT_STUDIES:
                del PERSISTENT_STUDIES[study_id]
            status = "Study reset"
        
        # Check if study exists in memory
        if study_id in PERSISTENT_STUDIES and not reset:
            study = PERSISTENT_STUDIES[study_id]
            status = f"Loaded from memory"
        # Try loading from disk
        elif save_path.exists() and not reset:
            try:
                with open(save_path, 'rb') as f:
                    study = pickle.load(f)
                PERSISTENT_STUDIES[study_id] = study
                status = f"Loaded from disk"
            except Exception as e:
                print(f"Failed to load study: {e}")
                study = None
                status = "Load failed, creating new"
        else:
            study = None
            status = "Creating new study"
        
        # Create new study if needed
        if study is None:
            # Configure sampler
            if sampler == "TPE":
                sampler_obj = TPESampler(seed=42)
            elif sampler == "Random":
                sampler_obj = optuna.samplers.RandomSampler(seed=42)
            else:
                sampler_obj = optuna.samplers.CmaEsSampler(seed=42)
            
            # Configure pruner
            if pruner == "MedianPruner":
                pruner_obj = optuna.pruners.MedianPruner()
            elif pruner == "HyperbandPruner":
                pruner_obj = optuna.pruners.HyperbandPruner()
            else:
                pruner_obj = None
            
            # Create study
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                sampler=sampler_obj,
                pruner=pruner_obj
            )
            
            PERSISTENT_STUDIES[study_id] = study
            
            # Save to disk
            with open(save_path, 'wb') as f:
                pickle.dump(study, f)
        
        # Get study statistics
        n_trials = len(study.trials)
        try:
            best_value = study.best_value if study.best_trial else 0.0
        except ValueError:
            # No trials completed yet
            best_value = 0.0
        
        # Wrap study for passing
        wrapped_study = StudyWrapper(study)
        
        # Auto-save after each use
        with open(save_path, 'wb') as f:
            pickle.dump(study, f)
        
        return (wrapped_study, n_trials, float(best_value), status)

class OptimizerSuggestNode:
    """
    Parameter proposer - asks the study for next parameters to test.
    Supports dynamic parameter definition.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "study": ("STUDY",),
                "param_config": ("STRING", {
                    "default": json.dumps([
                        {"name": "cfg", "type": "float", "low": 1.0, "high": 20.0},
                        {"name": "steps", "type": "int", "low": 10, "high": 50},
                        {"name": "sampler", "type": "categorical", "choices": ["euler", "dpmpp_2m", "ddim"]}
                    ], indent=2),
                    "multiline": True
                }),
            },
            "optional": {
                "use_best_trial": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("TRIAL", "FLOAT", "INT", "STRING", "FLOAT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("trial", "param1", "param2", "param3", "param4", "param5", "param6", "params_json")
    FUNCTION = "suggest_parameters"
    CATEGORY = "Optimization/Core"
    
    def suggest_parameters(self, study, param_config, use_best_trial=False):
        """Suggest next parameters or use best trial"""
        
        # Unwrap study
        if isinstance(study, StudyWrapper):
            study = study.study
        
        # Parse parameter configuration
        try:
            params_def = json.loads(param_config)
        except json.JSONDecodeError:
            raise ValueError("Invalid parameter configuration JSON")
        
        # Use best trial if requested and available
        if use_best_trial and study.best_trial:
            trial = study.best_trial
            params = trial.params
        else:
            # Get new trial
            trial = study.ask()
            
            # Suggest parameters based on configuration
            params = {}
            for param in params_def:
                name = param["name"]
                param_type = param["type"]
                
                if param_type == "float":
                    params[name] = trial.suggest_float(
                        name, 
                        param.get("low", 0.0), 
                        param.get("high", 1.0),
                        step=param.get("step", None),
                        log=param.get("log", False)
                    )
                elif param_type == "int":
                    params[name] = trial.suggest_int(
                        name,
                        param.get("low", 0),
                        param.get("high", 100),
                        step=param.get("step", 1),
                        log=param.get("log", False)
                    )
                elif param_type == "categorical":
                    params[name] = trial.suggest_categorical(
                        name,
                        param.get("choices", ["option1", "option2"])
                    )
        
        # Wrap trial
        wrapped_trial = TrialWrapper(trial, id(study))
        
        # Extract up to 6 parameters for individual outputs
        param_values = list(params.values())[:6]
        while len(param_values) < 6:
            param_values.append("")  # Pad with empty strings
        
        # Convert all to appropriate types
        output_params = []
        for i, val in enumerate(param_values):
            if isinstance(val, float):
                output_params.append(float(val))
            elif isinstance(val, int):
                output_params.append(int(val))
            else:
                output_params.append(str(val))
        
        # Add JSON representation for complex workflows
        params_json = json.dumps(params)
        
        return (wrapped_trial, *output_params, params_json)

class ScoringNode:
    """
    Image quality evaluator - computes similarity scores.
    Judges how well the generated image matches the target.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "metric": (["combined", "dreamsim", "lpips", "mse", "ssim"],),
                "weight_dreamsim": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "weight_lpips": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "weight_mse": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "normalize": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("score", "dreamsim_score", "lpips_score", "mse_score", "details")
    FUNCTION = "compute_score"
    CATEGORY = "Optimization/Metrics"
    
    def __init__(self):
        """Initialize metric models"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize DreamSim
        if DREAMSIM_AVAILABLE:
            self.dreamsim_model, _ = dreamsim(pretrained=True, device=self.device)
            self.dreamsim_model.eval()
        else:
            self.dreamsim_model = None
        
        # Initialize LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None
    
    def compute_score(self, generated_image, target_image, metric, 
                     weight_dreamsim, weight_lpips, weight_mse, normalize=True):
        """Compute similarity score between images"""
        
        # Convert images to tensors if needed
        if not isinstance(generated_image, torch.Tensor):
            generated_image = torch.from_numpy(generated_image)
        if not isinstance(target_image, torch.Tensor):
            target_image = torch.from_numpy(target_image)
        
        # Ensure correct shape [B, H, W, C] -> [B, C, H, W]
        if generated_image.dim() == 3:
            generated_image = generated_image.unsqueeze(0)
        if target_image.dim() == 3:
            target_image = target_image.unsqueeze(0)
        
        if generated_image.shape[-1] == 3:
            generated_image = generated_image.permute(0, 3, 1, 2)
        if target_image.shape[-1] == 3:
            target_image = target_image.permute(0, 3, 1, 2)
        
        # Move to device
        generated_image = generated_image.to(self.device)
        target_image = target_image.to(self.device)
        
        # Ensure same size
        if generated_image.shape != target_image.shape:
            target_image = torch.nn.functional.interpolate(
                target_image, 
                size=generated_image.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize to [-1, 1] for LPIPS/DreamSim
        if normalize:
            gen_norm = generated_image * 2 - 1
            tgt_norm = target_image * 2 - 1
        else:
            gen_norm = generated_image
            tgt_norm = target_image
        
        # Compute individual metrics
        scores = {}
        
        # DreamSim
        if self.dreamsim_model and metric in ["combined", "dreamsim"]:
            with torch.no_grad():
                dreamsim_score = self.dreamsim_model(gen_norm, tgt_norm).item()
            scores["dreamsim"] = 1.0 - dreamsim_score  # Convert distance to similarity
        else:
            scores["dreamsim"] = 0.0
        
        # LPIPS
        if self.lpips_model and metric in ["combined", "lpips"]:
            with torch.no_grad():
                lpips_score = self.lpips_model(gen_norm, tgt_norm).item()
            scores["lpips"] = 1.0 - lpips_score  # Convert distance to similarity
        else:
            scores["lpips"] = 0.0
        
        # MSE
        if metric in ["combined", "mse"]:
            mse = torch.nn.functional.mse_loss(generated_image, target_image).item()
            scores["mse"] = 1.0 / (1.0 + mse)  # Convert to similarity
        else:
            scores["mse"] = 0.0
        
        # SSIM (simplified)
        if metric == "ssim":
            # Simplified SSIM calculation
            mu1 = generated_image.mean()
            mu2 = target_image.mean()
            sigma1 = generated_image.std()
            sigma2 = target_image.std()
            sigma12 = ((generated_image - mu1) * (target_image - mu2)).mean()
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
            scores["ssim"] = ssim.item()
        
        # Compute combined score
        if metric == "combined":
            total_weight = weight_dreamsim + weight_lpips + weight_mse
            if total_weight > 0:
                final_score = (
                    scores["dreamsim"] * weight_dreamsim +
                    scores["lpips"] * weight_lpips +
                    scores["mse"] * weight_mse
                ) / total_weight
            else:
                final_score = 0.0
        else:
            final_score = scores.get(metric, 0.0)
        
        # Create details string
        details = json.dumps({
            "dreamsim": round(scores["dreamsim"], 4),
            "lpips": round(scores["lpips"], 4),
            "mse": round(scores["mse"], 4),
            "final": round(final_score, 4)
        })
        
        return (
            float(final_score),
            float(scores["dreamsim"]),
            float(scores["lpips"]),
            float(scores["mse"]),
            details
        )

class OptimizerTellNode:
    """
    Feedback provider - tells the study the result of the trial.
    Closes the optimization loop.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "study": ("STUDY",),
                "trial": ("TRIAL",),
                "score": ("FLOAT",),
            },
            "optional": {
                "save_intermediate": ("BOOLEAN", {"default": True}),
                "prune_trial": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STUDY", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("study", "status", "best_score", "trial_number")
    FUNCTION = "complete_trial"
    CATEGORY = "Optimization/Core"
    
    def complete_trial(self, study, trial, score, save_intermediate=True, prune_trial=False):
        """Complete the trial and update the study"""
        
        # Unwrap objects
        if isinstance(study, StudyWrapper):
            study_obj = study.study
        else:
            study_obj = study
        
        if isinstance(trial, TrialWrapper):
            trial_obj = trial.trial
        else:
            trial_obj = trial
        
        # Handle pruning
        if prune_trial:
            trial_obj.report(score, step=0)
            if trial_obj.should_prune():
                study_obj.tell(trial_obj, state=optuna.trial.TrialState.PRUNED)
                status = f"Trial {trial_obj.number} pruned"
            else:
                study_obj.tell(trial_obj, score)
                status = f"Trial {trial_obj.number} completed"
        else:
            # Complete the trial
            study_obj.tell(trial_obj, score)
            status = f"Trial {trial_obj.number} completed with score: {score:.4f}"
        
        # Save intermediate results if requested
        if save_intermediate:
            study_id = hashlib.md5(study_obj.study_name.encode()).hexdigest()[:8]
            save_path = get_studies_dir() / f"{study_obj.study_name}_{study_id}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(study_obj, f)
        
        # Get best score
        best_score = study_obj.best_value if study_obj.best_trial else 0.0
        
        # Update status with improvement info
        if study_obj.best_trial and study_obj.best_trial.number == trial_obj.number:
            status += " (NEW BEST!)"
        
        return (study, status, float(best_score), int(trial_obj.number))

class TrialPassthroughNode:
    """
    Helper node to pass TRIAL objects through parts of the graph
    that don't need to interact with them.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trial": ("TRIAL",),
            },
            "optional": {
                "any_input": ("*",),  # Accept any input to pass through
            }
        }
    
    RETURN_TYPES = ("TRIAL", "*")
    RETURN_NAMES = ("trial", "passthrough")
    FUNCTION = "passthrough"
    CATEGORY = "Optimization/Utility"
    
    def passthrough(self, trial, any_input=None):
        """Pass trial object through unchanged"""
        return (trial, any_input)

class BestParametersNode:
    """
    Extracts the best parameters found so far from a study.
    Useful for applying the best configuration.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "study": ("STUDY",),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "FLOAT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("best_params_json", "param1", "param2", "param3", "param4", "param5", "param6")
    FUNCTION = "get_best_parameters"
    CATEGORY = "Optimization/Utility"
    
    def get_best_parameters(self, study):
        """Extract best parameters from study"""
        
        # Unwrap study
        if isinstance(study, StudyWrapper):
            study = study.study
        
        if not study.best_trial:
            # Return defaults if no trials yet
            return ("No trials yet", 0.0, 0, 0.0, 0, "", "")
        
        best_params = study.best_trial.params
        params_json = json.dumps(best_params, indent=2)
        
        # Extract individual parameters
        param_values = list(best_params.values())[:6]
        while len(param_values) < 6:
            param_values.append("")
        
        # Convert to appropriate types
        output_params = []
        for val in param_values:
            if isinstance(val, float):
                output_params.append(float(val))
            elif isinstance(val, int):
                output_params.append(int(val))
            else:
                output_params.append(str(val))
        
        return (params_json, *output_params)

class StudyVisualizerNode:
    """
    Generates visualization of optimization progress.
    Creates plots showing parameter importance and optimization history.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "study": ("STUDY",),
                "plot_type": (["history", "importance", "parallel", "slice"],),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("plot", "analysis")
    FUNCTION = "visualize_study"
    CATEGORY = "Optimization/Analysis"
    
    def visualize_study(self, study, plot_type, width, height):
        """Generate visualization of study progress"""
        
        # Import here to avoid dependency if not used
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            # Return blank image if matplotlib not available
            blank = np.ones((height, width, 3), dtype=np.float32)
            return (blank, "Matplotlib not installed")
        
        # Unwrap study
        if isinstance(study, StudyWrapper):
            study = study.study
        
        if len(study.trials) == 0:
            blank = np.ones((height, width, 3), dtype=np.float32)
            return (blank, "No trials to visualize")
        
        # Create figure
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        
        if plot_type == "history":
            # Plot optimization history
            ax = fig.add_subplot(111)
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if trials:
                x = [t.number for t in trials]
                y = [t.value for t in trials]
                ax.plot(x, y, 'b-', alpha=0.5, label='Trial values')
                
                # Add best value line
                best_vals = []
                current_best = float('inf') if study.direction == optuna.study.StudyDirection.MINIMIZE else float('-inf')
                for val in y:
                    if study.direction == optuna.study.StudyDirection.MINIMIZE:
                        current_best = min(current_best, val)
                    else:
                        current_best = max(current_best, val)
                    best_vals.append(current_best)
                ax.plot(x, best_vals, 'r-', linewidth=2, label='Best value')
                
                ax.set_xlabel('Trial Number')
                ax.set_ylabel('Objective Value')
                ax.set_title('Optimization History')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        elif plot_type == "importance":
            # Plot parameter importance
            try:
                from optuna.importance import get_param_importances
                importance = get_param_importances(study)
                
                ax = fig.add_subplot(111)
                params = list(importance.keys())
                values = list(importance.values())
                
                y_pos = np.arange(len(params))
                ax.barh(y_pos, values)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(params)
                ax.set_xlabel('Importance')
                ax.set_title('Parameter Importance')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f'Cannot compute importance:\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == "parallel":
            # Parallel coordinates plot
            ax = fig.add_subplot(111)
            
            # Get completed trials
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if trials and trials[0].params:
                param_names = list(trials[0].params.keys())
                
                # Normalize parameters to [0, 1]
                normalized_data = []
                for trial in trials:
                    row = []
                    for param in param_names:
                        val = trial.params.get(param, 0)
                        if isinstance(val, str):
                            # Handle categorical
                            unique_vals = list(set(t.params.get(param, '') for t in trials))
                            val = unique_vals.index(val) / max(1, len(unique_vals) - 1)
                        row.append(val)
                    normalized_data.append(row)
                
                # Plot lines
                x = np.arange(len(param_names))
                for i, row in enumerate(normalized_data):
                    alpha = 0.1 if i < len(normalized_data) - 10 else 0.5
                    ax.plot(x, row, alpha=alpha)
                
                ax.set_xticks(x)
                ax.set_xticklabels(param_names, rotation=45)
                ax.set_ylabel('Normalized Value')
                ax.set_title('Parallel Coordinates')
                ax.grid(True, alpha=0.3)
        
        else:  # slice
            # Plot slice of objective function
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Slice plot not yet implemented',
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Convert to float32 [0, 1]
        image = buf.astype(np.float32) / 255.0
        
        # Generate analysis text
        analysis = f"Study: {study.study_name}\n"
        analysis += f"Trials: {len(study.trials)}\n"
        analysis += f"Best value: {study.best_value if study.best_trial else 'N/A'}\n"
        if study.best_trial:
            analysis += f"Best params: {json.dumps(study.best_trial.params, indent=2)}"
        
        return (image, analysis)

class SamplerAdapter:
    """
    Converts string sampler names to ComfyUI's expected combo type.
    Bridges the gap between optimizer output and KSampler input.
    """
    
    SAMPLERS = [
        "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
        "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
        "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
        "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp",
        "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
        "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "ddim", "uni_pc", "uni_pc_bh2"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_string": ("STRING",),
                "fallback": (cls.SAMPLERS,),
            }
        }
    
    RETURN_TYPES = (cls.SAMPLERS,)
    RETURN_NAMES = ("sampler_name",)
    FUNCTION = "convert"
    CATEGORY = "Optimization/Utility"
    
    def convert(self, sampler_string, fallback):
        """Convert string to valid sampler name"""
        # Clean the input string
        sampler_clean = sampler_string.strip().lower()
        
        # Direct match
        if sampler_clean in self.SAMPLERS:
            return (sampler_clean,)
        
        # Try to find a partial match
        for sampler in self.SAMPLERS:
            if sampler_clean in sampler or sampler in sampler_clean:
                return (sampler,)
        
        # Use fallback if no match found
        return (fallback,)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "OptimizerStateNode": OptimizerStateNode,
    "OptimizerSuggestNode": OptimizerSuggestNode,
    "ScoringNode": ScoringNode,
    "OptimizerTellNode": OptimizerTellNode,
    "TrialPassthroughNode": TrialPassthroughNode,
    "BestParametersNode": BestParametersNode,
    "StudyVisualizerNode": StudyVisualizerNode,
    "SamplerAdapter": SamplerAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OptimizerStateNode": "🧠 Optimizer State",
    "OptimizerSuggestNode": "💡 Suggest Parameters",
    "ScoringNode": "📊 Score Images",
    "OptimizerTellNode": "✍️ Complete Trial",
    "TrialPassthroughNode": "🔄 Pass Trial",
    "BestParametersNode": "🏆 Best Parameters",
    "StudyVisualizerNode": "📈 Visualize Study",
    "SamplerAdapter": "🔄 Sampler Adapter",
}