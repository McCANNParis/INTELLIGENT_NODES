"""
Post-Processing Nodes for Bayesian Optimization
These nodes calculate similarity AFTER image generation to avoid circular dependencies
"""

import os
import json
import pickle
import numpy as np
import torch
from typing import Tuple, Optional
from PIL import Image as PILImage


class BayesianIterationManager:
    """Manages iteration state and parameters without circular dependencies"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
                "iteration_trigger": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
            "optional": {
                "previous_score": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("guidance", "steps", "seed", "is_complete", "status")
    FUNCTION = "manage_iteration"
    CATEGORY = "Bayesian Optimization/Post-Process"
    
    def manage_iteration(self, config, iteration_trigger, previous_score=0.5):
        """Manage iteration without creating circular dependencies"""
        import pickle
        import os
        import numpy as np
        
        # State file path
        state_file = "/tmp/bayesian_iteration_state.pkl"
        
        # Load or initialize state
        if os.path.exists(state_file) and iteration_trigger > 1:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
        else:
            state = {
                'iteration': 1,
                'history': [],
                'scores': [],
                'parameters': []
            }
        
        current_iteration = state['iteration']
        
        # Check if optimization is complete
        is_complete = current_iteration >= config.get('total_iterations', 20)
        
        if is_complete:
            status = f"Optimization complete after {current_iteration} iterations"
            # Return last best parameters
            if state['parameters']:
                best_idx = np.argmax(state['scores'])
                best_params = state['parameters'][best_idx]
                return (
                    float(best_params.get('guidance', 3.5)),
                    int(best_params.get('steps', 20)),
                    int(best_params.get('seed', 42)),
                    True,
                    status
                )
        
        # Generate new parameters based on history
        if current_iteration == 1 or not state['history']:
            # First iteration or no history - use initial sampling
            guidance = float(np.random.uniform(
                config.get('guidance_min', 1),
                config.get('guidance_max', 7)
            ))
            steps = int(np.random.randint(
                config.get('steps_min', 15),
                config.get('steps_max', 30)
            ))
            seed = int(np.random.randint(0, 1000000))
        else:
            # Use Bayesian optimization to suggest next parameters
            try:
                from skopt.acquisition import gaussian_ei
                from sklearn.gaussian_process import GaussianProcessRegressor
                
                # Prepare data
                X = np.array([[p['guidance'], p['steps']] for p in state['parameters']])
                y = np.array(state['scores'])
                
                # Fit GP model
                gp = GaussianProcessRegressor(alpha=1e-6, normalize_y=True)
                gp.fit(X, y)
                
                # Generate candidates
                n_candidates = 100
                candidates = []
                for _ in range(n_candidates):
                    cand_guidance = np.random.uniform(
                        config.get('guidance_min', 1),
                        config.get('guidance_max', 7)
                    )
                    cand_steps = np.random.randint(
                        config.get('steps_min', 15),
                        config.get('steps_max', 30)
                    )
                    candidates.append([cand_guidance, cand_steps])
                
                candidates = np.array(candidates)
                
                # Calculate acquisition values
                acq_values = gaussian_ei(candidates, gp, np.min(y))
                
                # Select best candidate
                best_idx = np.argmax(acq_values)
                guidance = float(candidates[best_idx][0])
                steps = int(candidates[best_idx][1])
                seed = int(np.random.randint(0, 1000000))
            except ImportError:
                # Fallback to random if scikit-optimize not available
                guidance = float(np.random.uniform(
                    config.get('guidance_min', 1),
                    config.get('guidance_max', 7)
                ))
                steps = int(np.random.randint(
                    config.get('steps_min', 15),
                    config.get('steps_max', 30)
                ))
                seed = int(np.random.randint(0, 1000000))
        
        # Update state for next iteration
        if iteration_trigger > state.get('last_trigger', 0):
            # New iteration triggered
            state['parameters'].append({
                'guidance': guidance,
                'steps': steps,
                'seed': seed
            })
            if previous_score > 0:
                state['scores'].append(previous_score)
            state['iteration'] = current_iteration + 1
            state['last_trigger'] = iteration_trigger
            
            # Save state
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
        
        status = f"Iteration {current_iteration}/{config.get('total_iterations', 20)}"
        
        return (
            float(guidance),
            int(steps),
            int(seed),
            False,
            status
        )


class PostGenerationScorer:
    """Calculate similarity after generation without feedback loops"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "save_comparison": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("similarity_score", "report", "comparison_image")
    FUNCTION = "calculate_similarity"
    CATEGORY = "Bayesian Optimization/Post-Process"
    OUTPUT_NODE = True
    
    def calculate_similarity(self, generated_image, target_image, save_comparison):
        """Calculate similarity and optionally save comparison"""
        import numpy as np
        import torch
        from PIL import Image as PILImage
        import os
        
        # Convert tensors to numpy
        if torch.is_tensor(generated_image):
            gen_np = generated_image.cpu().numpy()
            if gen_np.ndim == 4:
                gen_np = gen_np[0]
            if gen_np.shape[0] in [3, 4]:
                gen_np = np.transpose(gen_np, (1, 2, 0))
        else:
            gen_np = np.array(generated_image)
        
        if torch.is_tensor(target_image):
            tgt_np = target_image.cpu().numpy()
            if tgt_np.ndim == 4:
                tgt_np = tgt_np[0]
            if tgt_np.shape[0] in [3, 4]:
                tgt_np = np.transpose(tgt_np, (1, 2, 0))
        else:
            tgt_np = np.array(target_image)
        
        # Ensure same shape
        if gen_np.shape != tgt_np.shape:
            # Resize generated to match target
            h, w = tgt_np.shape[:2]
            gen_pil = PILImage.fromarray((np.clip(gen_np, 0, 1) * 255).astype(np.uint8))
            gen_pil = gen_pil.resize((w, h), PILImage.Resampling.LANCZOS)
            gen_np = np.array(gen_pil).astype(np.float32) / 255.0
        
        # Calculate MSE
        mse = np.mean((gen_np - tgt_np) ** 2)
        similarity = 1.0 / (1.0 + mse * 10)
        
        # Create comparison image
        comparison = np.concatenate([tgt_np, gen_np], axis=1)
        
        # Generate report
        report = f"Similarity Score: {similarity:.4f}\n"
        report += f"MSE: {mse:.6f}\n"
        
        # Load iteration state if exists
        state_file = "/tmp/bayesian_iteration_state.pkl"
        if os.path.exists(state_file):
            try:
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                    iteration = state.get('iteration', 1)
                    report += f"Iteration: {iteration}\n"
                    
                    if state.get('scores'):
                        best_score = max(state['scores'])
                        avg_score = np.mean(state['scores'])
                        report += f"Best Score So Far: {best_score:.4f}\n"
                        report += f"Average Score: {avg_score:.4f}\n"
            except:
                pass
        
        # Save comparison if requested
        if save_comparison:
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
            except:
                output_dir = "/workspace/ComfyUI/output"
            
            comp_dir = os.path.join(output_dir, "bayesian_comparisons")
            os.makedirs(comp_dir, exist_ok=True)
            
            # Save with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}_score_{similarity:.3f}.png"
            filepath = os.path.join(comp_dir, filename)
            
            comp_pil = PILImage.fromarray((np.clip(comparison, 0, 1) * 255).astype(np.uint8))
            comp_pil.save(filepath)
            report += f"\nComparison saved to: {filename}"
        
        # Convert comparison back to tensor format for output
        if comparison.ndim == 3:
            comparison = np.expand_dims(comparison, 0)
        comparison_tensor = torch.from_numpy(comparison)
        
        return (float(similarity), report, comparison_tensor)


class IterationTrigger:
    """Simple counter to trigger iterations without loops"""
    
    _counter = 0
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "increment": ("BOOLEAN", {"default": True}),
                "reset": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("counter", "status")
    FUNCTION = "update_counter"
    CATEGORY = "Bayesian Optimization/Post-Process"
    
    def update_counter(self, increment, reset):
        """Update counter for triggering iterations"""
        if reset:
            IterationTrigger._counter = 0
            return (0, "Counter reset")
        
        if increment:
            IterationTrigger._counter += 1
        
        return (IterationTrigger._counter, f"Counter: {IterationTrigger._counter}")


class AutoQueueManager:
    """Manages automatic queuing without circular dependencies"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_iterations": ("INT", {"default": 20, "min": 1, "max": 100}),
                "current_iteration": ("INT", {"default": 1, "min": 0, "max": 100}),
                "auto_queue": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "STRING", "INT")
    RETURN_NAMES = ("should_continue", "status", "next_iteration")
    FUNCTION = "manage_queue"
    CATEGORY = "Bayesian Optimization/Post-Process"
    OUTPUT_NODE = True
    
    def manage_queue(self, total_iterations, current_iteration, auto_queue):
        """Determine if optimization should continue"""
        should_continue = current_iteration < total_iterations and auto_queue
        next_iteration = current_iteration + 1 if should_continue else current_iteration
        
        if should_continue:
            status = f"Queuing iteration {next_iteration}/{total_iterations}"
        elif current_iteration >= total_iterations:
            status = f"Optimization complete: {current_iteration}/{total_iterations} iterations"
        else:
            status = "Auto-queue disabled"
        
        return (should_continue, status, next_iteration)


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "BayesianIterationManager": BayesianIterationManager,
    "PostGenerationScorer": PostGenerationScorer,
    "IterationTrigger": IterationTrigger,
    "AutoQueueManager": AutoQueueManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BayesianIterationManager": "Iteration Manager (No Loop)",
    "PostGenerationScorer": "Post-Generation Scorer",
    "IterationTrigger": "Iteration Trigger",
    "AutoQueueManager": "Auto Queue Manager",
}