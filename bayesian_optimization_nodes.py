"""
Bayesian Optimization Nodes for ComfyUI
Optimizes Flux Dev parameters using Gaussian Process regression
"""

import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from PIL import Image
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

# Try to import similarity metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

class BayesianOptimizerConfig:
    """Configuration node for Bayesian optimization parameters"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "fixed_prompt": ("STRING", {"multiline": True}),
                "n_iterations": ("INT", {"default": 50, "min": 10, "max": 200}),
                "n_initial_points": ("INT", {"default": 10, "min": 5, "max": 50}),
                "cfg_min": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0}),
                "cfg_max": ("FLOAT", {"default": 7.0, "min": 0.1, "max": 20.0}),
                "steps_min": ("INT", {"default": 20, "min": 1, "max": 150}),
                "steps_max": ("INT", {"default": 50, "min": 1, "max": 150}),
                "lora_weight_min": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0}),
                "lora_weight_max": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0}),
                "similarity_metric": (["LPIPS", "CLIP", "MSE", "SSIM", "Combined"],),
                "optimization_seed": ("INT", {"default": 42}),
            }
        }
    
    RETURN_TYPES = ("BAYESIAN_CONFIG",)
    FUNCTION = "create_config"
    CATEGORY = "Bayesian Optimization"
    
    def create_config(self, target_image, fixed_prompt, n_iterations, n_initial_points,
                     cfg_min, cfg_max, steps_min, steps_max, 
                     lora_weight_min, lora_weight_max,
                     similarity_metric, optimization_seed):
        
        # Create parameter space
        if SKOPT_AVAILABLE:
            space = [
                Real(cfg_min, cfg_max, name='cfg'),
                Integer(steps_min, steps_max, name='steps'),
                Real(lora_weight_min, lora_weight_max, name='lora_weight')
            ]
        else:
            space = {
                'cfg': (cfg_min, cfg_max),
                'steps': (steps_min, steps_max),
                'lora_weight': (lora_weight_min, lora_weight_max)
            }
        
        config = {
            "target_image": target_image,
            "fixed_prompt": fixed_prompt,
            "n_iterations": n_iterations,
            "n_initial_points": n_initial_points,
            "space": space,
            "similarity_metric": similarity_metric,
            "optimization_seed": optimization_seed,
            "history": [],
            "best_params": None,
            "best_score": float('-inf'),
            "iteration": 0
        }
        
        return (config,)

class BayesianParameterSampler:
    """Samples parameters using Bayesian optimization"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("BAYESIAN_CONFIG",),
                "similarity_score": ("FLOAT", {"default": 0.0}),
                "is_first_run": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "BAYESIAN_CONFIG", "BOOLEAN")
    RETURN_NAMES = ("cfg", "steps", "lora_weight", "updated_config", "optimization_complete")
    FUNCTION = "sample_parameters"
    CATEGORY = "Bayesian Optimization"
    
    def __init__(self):
        self.optimizer = None
        self.current_params = None
    
    def sample_parameters(self, config, similarity_score, is_first_run):
        # Update history if not first run
        if not is_first_run and self.current_params is not None:
            config["history"].append({
                "params": self.current_params,
                "score": similarity_score,
                "iteration": config["iteration"]
            })
            
            # Update best parameters
            if similarity_score > config["best_score"]:
                config["best_score"] = similarity_score
                config["best_params"] = self.current_params.copy()
        
        # Check if optimization is complete
        if config["iteration"] >= config["n_iterations"]:
            if config["best_params"] is not None:
                return (
                    config["best_params"]["cfg"],
                    int(config["best_params"]["steps"]),
                    config["best_params"]["lora_weight"],
                    config,
                    True
                )
            else:
                # Return defaults if no best params found
                return (3.5, 30, 0.8, config, True)
        
        # Sample new parameters
        if SKOPT_AVAILABLE:
            params = self._sample_skopt(config)
        else:
            params = self._sample_basic(config)
        
        self.current_params = {
            "cfg": params[0],
            "steps": params[1],
            "lora_weight": params[2]
        }
        
        config["iteration"] += 1
        
        return (
            float(params[0]),  # cfg
            int(params[1]),    # steps
            float(params[2]),  # lora_weight
            config,
            False
        )
    
    def _sample_skopt(self, config):
        """Sample using scikit-optimize"""
        if config["iteration"] < config["n_initial_points"]:
            # Use Sobol sequence for initial exploration
            from skopt.sampler import Sobol
            sampler = Sobol()
            return sampler.generate(config["space"], 1)[0]
        else:
            # Use Gaussian Process for informed sampling
            X = []
            y = []
            for entry in config["history"]:
                X.append([
                    entry["params"]["cfg"],
                    entry["params"]["steps"],
                    entry["params"]["lora_weight"]
                ])
                y.append(-entry["score"])  # Minimize negative score
            
            # Fit GP and suggest next point
            from skopt.learning import GaussianProcessRegressor
            gpr = GaussianProcessRegressor()
            gpr.fit(X, y)
            
            # Find next point to evaluate
            from skopt.acquisition import gaussian_ei
            next_x = gaussian_ei(X, gpr, y_min=min(y))
            
            return next_x
    
    def _sample_basic(self, config):
        """Basic random sampling fallback"""
        np.random.seed(config["optimization_seed"] + config["iteration"])
        
        cfg_range = config["space"]["cfg"]
        steps_range = config["space"]["steps"]
        lora_range = config["space"]["lora_weight"]
        
        if config["iteration"] < config["n_initial_points"]:
            # Uniform sampling for initial points
            cfg = np.random.uniform(cfg_range[0], cfg_range[1])
            steps = np.random.randint(steps_range[0], steps_range[1] + 1)
            lora_weight = np.random.uniform(lora_range[0], lora_range[1])
        else:
            # Sample around best point with decreasing variance
            if config["best_params"] is not None:
                variance = 0.3 * (1 - config["iteration"] / config["n_iterations"])
                cfg = np.clip(
                    config["best_params"]["cfg"] + np.random.normal(0, variance),
                    cfg_range[0], cfg_range[1]
                )
                steps = np.clip(
                    int(config["best_params"]["steps"] + np.random.normal(0, variance * 10)),
                    steps_range[0], steps_range[1]
                )
                lora_weight = np.clip(
                    config["best_params"]["lora_weight"] + np.random.normal(0, variance),
                    lora_range[0], lora_range[1]
                )
            else:
                # Random if no best params yet
                cfg = np.random.uniform(cfg_range[0], cfg_range[1])
                steps = np.random.randint(steps_range[0], steps_range[1] + 1)
                lora_weight = np.random.uniform(lora_range[0], lora_range[1])
        
        return [cfg, steps, lora_weight]

class ImageSimilarityScorer:
    """Calculates similarity between generated and target images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "config": ("BAYESIAN_CONFIG",),
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("similarity_score",)
    FUNCTION = "calculate_similarity"
    CATEGORY = "Bayesian Optimization"
    
    def __init__(self):
        self.lpips_model = None
        self.clip_model = None
        self.clip_preprocess = None
    
    def calculate_similarity(self, generated_image, config):
        target_image = config["target_image"]
        metric = config["similarity_metric"]
        
        # Ensure images are in the same format
        if isinstance(generated_image, torch.Tensor):
            gen_tensor = generated_image
        else:
            gen_tensor = torch.from_numpy(generated_image).float()
        
        if isinstance(target_image, torch.Tensor):
            target_tensor = target_image
        else:
            target_tensor = torch.from_numpy(target_image).float()
        
        # Calculate similarity based on chosen metric
        if metric == "LPIPS" and LPIPS_AVAILABLE:
            score = self._calculate_lpips(gen_tensor, target_tensor)
        elif metric == "CLIP" and CLIP_AVAILABLE:
            score = self._calculate_clip(gen_tensor, target_tensor)
        elif metric == "MSE":
            score = self._calculate_mse(gen_tensor, target_tensor)
        elif metric == "SSIM":
            score = self._calculate_ssim(gen_tensor, target_tensor)
        elif metric == "Combined":
            scores = []
            scores.append(self._calculate_mse(gen_tensor, target_tensor))
            scores.append(self._calculate_ssim(gen_tensor, target_tensor))
            if LPIPS_AVAILABLE:
                scores.append(self._calculate_lpips(gen_tensor, target_tensor))
            score = np.mean(scores)
        else:
            # Fallback to MSE
            score = self._calculate_mse(gen_tensor, target_tensor)
        
        return (float(score),)
    
    def _calculate_lpips(self, img1, img2):
        """Calculate LPIPS perceptual similarity"""
        if self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net='alex')
        
        # Normalize images to [-1, 1]
        img1_norm = img1 * 2 - 1
        img2_norm = img2 * 2 - 1
        
        # Calculate LPIPS distance (lower is better, so we return 1 - distance)
        with torch.no_grad():
            distance = self.lpips_model(img1_norm, img2_norm)
        
        return 1.0 - float(distance.mean())
    
    def _calculate_clip(self, img1, img2):
        """Calculate CLIP embedding similarity"""
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        
        # Process images for CLIP
        img1_pil = Image.fromarray((img1.squeeze().numpy() * 255).astype(np.uint8))
        img2_pil = Image.fromarray((img2.squeeze().numpy() * 255).astype(np.uint8))
        
        img1_processed = self.clip_preprocess(img1_pil).unsqueeze(0)
        img2_processed = self.clip_preprocess(img2_pil).unsqueeze(0)
        
        # Get embeddings
        with torch.no_grad():
            img1_features = self.clip_model.encode_image(img1_processed)
            img2_features = self.clip_model.encode_image(img2_processed)
        
        # Calculate cosine similarity
        similarity = torch.cosine_similarity(img1_features, img2_features)
        
        return float(similarity)
    
    def _calculate_mse(self, img1, img2):
        """Calculate Mean Squared Error (inverted for similarity)"""
        mse = torch.mean((img1 - img2) ** 2)
        # Convert to similarity score (higher is better)
        return 1.0 / (1.0 + float(mse))
    
    def _calculate_ssim(self, img1, img2):
        """Calculate Structural Similarity Index"""
        # Simple SSIM implementation
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        
        sigma1_sq = torch.var(img1)
        sigma2_sq = torch.var(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(ssim)

class OptimizationVisualizer:
    """Visualizes optimization progress"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("BAYESIAN_CONFIG",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_progress"
    CATEGORY = "Bayesian Optimization"
    
    def visualize_progress(self, config):
        if not config["history"]:
            # Return placeholder if no history
            placeholder = np.ones((512, 512, 3), dtype=np.uint8) * 255
            return (torch.from_numpy(placeholder).float() / 255.0,)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Bayesian Optimization Progress - Iteration {config["iteration"]}', fontsize=16)
        
        iterations = [h["iteration"] for h in config["history"]]
        scores = [h["score"] for h in config["history"]]
        cfgs = [h["params"]["cfg"] for h in config["history"]]
        steps = [h["params"]["steps"] for h in config["history"]]
        lora_weights = [h["params"]["lora_weight"] for h in config["history"]]
        
        # Plot 1: Score over iterations
        axes[0, 0].plot(iterations, scores, 'b-', marker='o')
        if config["best_params"]:
            best_iter = next(i for i, h in enumerate(config["history"]) 
                            if h["score"] == config["best_score"])
            axes[0, 0].plot(iterations[best_iter], config["best_score"], 
                          'r*', markersize=15, label=f'Best: {config["best_score"]:.3f}')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Similarity Score')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Parameter evolution - CFG
        axes[0, 1].scatter(iterations, cfgs, c=scores, cmap='viridis', s=50)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('CFG Value')
        axes[0, 1].set_title('CFG Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Parameter evolution - Steps
        axes[1, 0].scatter(iterations, steps, c=scores, cmap='viridis', s=50)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].set_title('Steps Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Parameter evolution - LoRA weight
        axes[1, 1].scatter(iterations, lora_weights, c=scores, cmap='viridis', s=50)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('LoRA Weight')
        axes[1, 1].set_title('LoRA Weight Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=min(scores), vmax=max(scores)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.1)
        cbar.set_label('Similarity Score', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close()
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        
        return (img_tensor,)

class BayesianResultsExporter:
    """Exports optimization results"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("BAYESIAN_CONFIG",),
                "export_path": ("STRING", {"default": "bayesian_results"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_summary",)
    FUNCTION = "export_results"
    CATEGORY = "Bayesian Optimization"
    
    def export_results(self, config, export_path):
        # Create export directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = f"{export_path}_{timestamp}"
        os.makedirs(export_dir, exist_ok=True)
        
        # Save detailed results as JSON
        results = {
            "timestamp": timestamp,
            "total_iterations": config["iteration"],
            "best_score": float(config["best_score"]),
            "best_parameters": config["best_params"],
            "optimization_history": config["history"],
            "configuration": {
                "n_iterations": config["n_iterations"],
                "n_initial_points": config["n_initial_points"],
                "similarity_metric": config["similarity_metric"],
                "fixed_prompt": config["fixed_prompt"]
            }
        }
        
        with open(os.path.join(export_dir, "optimization_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary text
        summary = f"""
Bayesian Optimization Results
============================
Timestamp: {timestamp}
Total Iterations: {config["iteration"]}
Best Score: {config["best_score"]:.4f}

Best Parameters:
- CFG: {config["best_params"]["cfg"]:.2f}
- Steps: {config["best_params"]["steps"]}
- LoRA Weight: {config["best_params"]["lora_weight"]:.3f}

Fixed Prompt: {config["fixed_prompt"][:100]}...

Results saved to: {export_dir}
"""
        
        # Save summary
        with open(os.path.join(export_dir, "summary.txt"), 'w') as f:
            f.write(summary)
        
        # Plot final results
        if config["history"]:
            plt.figure(figsize=(10, 6))
            iterations = [h["iteration"] for h in config["history"]]
            scores = [h["score"] for h in config["history"]]
            
            plt.plot(iterations, scores, 'b-', marker='o', linewidth=2, markersize=6)
            plt.axhline(y=config["best_score"], color='r', linestyle='--', 
                       label=f'Best Score: {config["best_score"]:.4f}')
            
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Similarity Score', fontsize=12)
            plt.title('Bayesian Optimization Convergence', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(export_dir, "convergence_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        return (summary,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "BayesianOptimizerConfig": BayesianOptimizerConfig,
    "BayesianParameterSampler": BayesianParameterSampler,
    "ImageSimilarityScorer": ImageSimilarityScorer,
    "OptimizationVisualizer": OptimizationVisualizer,
    "BayesianResultsExporter": BayesianResultsExporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BayesianOptimizerConfig": "Bayesian Optimizer Config",
    "BayesianParameterSampler": "Bayesian Parameter Sampler",
    "ImageSimilarityScorer": "Image Similarity Scorer",
    "OptimizationVisualizer": "Optimization Visualizer",
    "BayesianResultsExporter": "Bayesian Results Exporter",
}