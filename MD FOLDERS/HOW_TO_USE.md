# How to Use Bayesian Optimization Nodes for ComfyUI

This guide explains how to use the Bayesian optimization custom nodes in ComfyUI to automatically optimize diffusion model parameters for better image generation results.

## Overview

These nodes use Bayesian optimization to intelligently search for optimal parameter combinations (CFG scale, sampling steps, sampler choice) by learning from previous generations. Instead of manually tuning parameters, the system learns which combinations work best for your specific use cases.

## Installation

1. **Navigate to ComfyUI custom nodes directory:**
   ```bash
   cd ComfyUI/custom_nodes
   git clone [repository-url] nodes_bayesian
   ```

2. **Install dependencies:**
   ```bash
   cd nodes_bayesian
   pip install -r requirements.txt
   ```

3. **Restart ComfyUI** to load the new nodes

## Available Nodes

### 1. Parameter Space Node
**Purpose:** Define the search space for optimization

**Inputs:**
- `cfg_min`: Minimum CFG scale value (default: 1.0)
- `cfg_max`: Maximum CFG scale value (default: 15.0)
- `steps_range`: Comma-separated list of step counts (e.g., "10,20,30,50")
- `samplers`: Comma-separated list of sampler names (e.g., "euler,dpmpp_2m,ddim")

**Output:** Parameter space configuration for the optimizer

### 2. Bayesian Optimizer Node
**Purpose:** Core optimization engine that suggests parameter combinations

**Inputs:**
- `model`: Your loaded diffusion model
- `positive`: Positive conditioning (prompt)
- `negative`: Negative conditioning
- `target_image`: Reference image for optimization (optional)
- `optimization_steps`: Number of optimization iterations (5-100)
- `acquisition_function`: Strategy for exploration ("EI", "UCB", or "PI")

**Outputs:**
- `optimal_cfg`: Best CFG scale found
- `optimal_steps`: Best step count found
- `optimal_sampler`: Best sampler found
- `history`: Optimization history for analysis

### 3. Metric Evaluator Node
**Purpose:** Evaluate generated images against targets or quality metrics

**Inputs:**
- `generated_image`: The image to evaluate
- `target_image`: Reference image for comparison
- `metrics`: Evaluation metrics to use (CLIP, LPIPS, SSIM, Aesthetic)

**Output:** Quality score (higher is better)

### 4. Bayesian Sampler Node
**Purpose:** Generate images using optimizer suggestions

**Inputs:**
- `model`: Diffusion model
- `positive`: Positive conditioning
- `negative`: Negative conditioning
- `latent_image`: Starting latent
- `bayes_optimizer`: Optimizer history from Bayesian Optimizer Node
- `iteration`: Current optimization iteration

**Outputs:**
- Generated image samples
- Used parameters for tracking

## Basic Workflow

### Simple Parameter Optimization

1. **Load your model and create prompts as usual**

2. **Add Parameter Space Node:**
   - Set your desired parameter ranges
   - Connect to Bayesian Optimizer

3. **Add Bayesian Optimizer Node:**
   - Connect your model, prompts, and parameter space
   - Set optimization steps (start with 10-20)
   - Choose acquisition function (EI recommended for beginners)

4. **Add Bayesian Sampler Node:**
   - Connect all inputs from your setup
   - Connect optimizer history from Bayesian Optimizer

5. **Add Metric Evaluator (optional):**
   - If you have a target image, use this to guide optimization
   - Otherwise, the system will optimize for general quality

6. **Run the workflow:**
   - The system will iteratively test parameters
   - Each run improves the parameter suggestions
   - Final outputs show optimal parameters found

## Advanced Usage

### Multi-Objective Optimization

Optimize for multiple goals simultaneously:
```
Target high quality AND fast generation:
- Use multiple Metric Evaluator nodes
- Weight different objectives in your workflow
```

### Transfer Learning

Use optimization results from one prompt/model on similar tasks:
```
1. Save optimization history from successful runs
2. Load as "previous_results" in new optimizations
3. System will start with learned knowledge
```

### Custom Metrics

Create custom evaluation metrics:
```python
# In your workflow, combine multiple evaluators:
- Aesthetic score for beauty
- CLIP score for prompt adherence  
- LPIPS for perceptual quality
```

## Tips and Best Practices

### Starting Out
- Begin with 10-20 optimization steps
- Use wider parameter ranges initially
- Narrow ranges as you learn what works

### Acquisition Functions
- **EI (Expected Improvement)**: Balanced exploration/exploitation
- **UCB (Upper Confidence Bound)**: More exploration
- **PI (Probability of Improvement)**: More exploitation

### Performance Optimization
- Cache model loads outside the optimization loop
- Use smaller test images during optimization
- Save successful parameter sets for reuse

### Common Workflows

**Style Transfer Optimization:**
1. Load style reference as target_image
2. Set metrics to ["CLIP", "LPIPS"]
3. Run 20-30 optimization steps
4. Use found parameters for final high-res generation

**Prompt Adherence Optimization:**
1. Focus on CLIP score metric
2. Test multiple samplers
3. Wider CFG range (1-20)
4. Save best parameters per prompt type

**Speed vs Quality:**
1. Include generation time in custom metric
2. Use smaller step ranges (10-30)
3. Prefer faster samplers in sampler list

## Troubleshooting

### Node doesn't appear in ComfyUI
- Check console for import errors
- Verify all dependencies installed
- Restart ComfyUI completely

### Optimization not improving
- Increase optimization steps
- Check parameter ranges aren't too narrow
- Try different acquisition function
- Verify metrics match your goals

### Memory issues
- Reduce optimization steps
- Use smaller test images
- Clear optimization history between runs

## Example Configurations

### Portrait Optimization
```
Parameter Space:
- cfg_min: 5.0, cfg_max: 12.0
- steps_range: "20,30,40"
- samplers: "dpmpp_2m,euler_ancestral"

Optimizer:
- optimization_steps: 15
- acquisition_function: "EI"
```

### Artistic Style
```
Parameter Space:
- cfg_min: 3.0, cfg_max: 20.0
- steps_range: "15,25,50,75"
- samplers: "euler,dpmpp_sde,ddim"

Optimizer:
- optimization_steps: 25
- acquisition_function: "UCB"
```

## Integration with Existing Workflows

These nodes integrate seamlessly with standard ComfyUI workflows:
1. Replace your KSampler with Bayesian Sampler
2. Add optimization nodes before sampling
3. Use outputs as you would normal sampler results
4. Combine with ControlNet, LoRA, etc. as usual

## Next Steps

- Experiment with different parameter ranges
- Create parameter presets for common tasks
- Share successful optimization results
- Contribute custom metrics to the project

For more technical details, see the CLAUDE.md file for implementation specifics.