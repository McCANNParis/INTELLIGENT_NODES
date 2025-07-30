# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ComfyUI custom nodes project focused on implementing Bayesian optimization for diffusion model parameters. The project aims to create nodes that can automatically optimize parameters like CFG scale, sampling steps, and sampler choice using Gaussian Process-based Bayesian optimization.

## Architecture

The planned architecture consists of four main node types:

1. **BayesianOptimizerNode**: Core optimization logic using Gaussian Processes
2. **ParameterSpaceNode**: Defines the search space for parameters
3. **MetricEvaluatorNode**: Evaluates generated images using various metrics (CLIP, LPIPS, SSIM, etc.)
4. **BayesianSamplerNode**: Integrates optimization suggestions into the sampling process

## Development Setup

### File Structure
ComfyUI custom nodes should be placed in the `ComfyUI/custom_nodes` folder. The project structure should be:
```
ComfyUI/custom_nodes/nodes_bayesian/
├── __init__.py          # Registers nodes with ComfyUI
├── nodes.py            # Main node implementations
├── bayesian_optimizer.py   # Bayesian optimization backend
├── metrics.py          # Image evaluation metrics
└── requirements.txt    # Dependencies (scikit-learn, etc.)
```

### Creating the Custom Nodes
1. **Navigate to ComfyUI custom_nodes directory:**
   ```bash
   cd ComfyUI/custom_nodes
   comfy node scaffold  # Optional: use scaffolding tool
   ```

2. **Node Structure Requirements:**
   - Class names must start with a capital letter
   - Must define `CATEGORY`, `INPUT_TYPES`, `RETURN_TYPES`, and `FUNCTION`
   - The `__init__.py` must contain `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`

### Example Implementation Structure

**__init__.py:**
```python
from .nodes import BayesianOptimizerNode, ParameterSpaceNode, MetricEvaluatorNode, BayesianSamplerNode

NODE_CLASS_MAPPINGS = {
    "BayesianOptimizer": BayesianOptimizerNode,
    "ParameterSpace": ParameterSpaceNode,
    "MetricEvaluator": MetricEvaluatorNode,
    "BayesianSampler": BayesianSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BayesianOptimizer": "Bayesian Optimizer",
    "ParameterSpace": "Parameter Space",
    "MetricEvaluator": "Metric Evaluator",
    "BayesianSampler": "Bayesian Sampler",
}
```

### Development Commands
- **Install dependencies:** Place in ComfyUI environment or use virtual env
- **Test nodes:** Restart ComfyUI after changes
- **Debug:** Check ComfyUI console for errors during node loading

## Key Technical Considerations

### ComfyUI Node API Requirements
- **CATEGORY**: Defines menu location in ComfyUI UI (e.g., "Bayesian/Optimization")
- **INPUT_TYPES**: Must return dict with "required" and optionally "optional" keys
- **RETURN_TYPES**: Tuple of output types (must match ComfyUI's type system)
- **FUNCTION**: String name of the method to execute
- **Node execution method**: Must return a tuple matching RETURN_TYPES

### Technical Implementation Details
- The implementation uses scikit-learn for Gaussian Process regression
- Integration with ComfyUI's node system requires following their API conventions
- The optimizer maintains history of observed parameters and scores for iterative improvement
- Support for multiple acquisition functions (Expected Improvement, Upper Confidence Bound, Probability of Improvement)

### ComfyUI Type System
Common types for this project:
- "MODEL": Diffusion model reference
- "CONDITIONING": Positive/negative prompts
- "IMAGE": Tensor images
- "LATENT": Latent space representations
- "FLOAT", "INT", "STRING": Basic types with widget controls
- Custom types: "BAYES_HISTORY", "PARAM_SPACE", "METRIC_FUNCTIONS"

## Testing Approach

When implementing, ensure to test:
- Parameter encoding/decoding for the Gaussian Process
- Acquisition function optimization
- Integration with ComfyUI's execution flow
- Metric computation accuracy
- Node loading in ComfyUI (check for import errors)
- Widget rendering and input validation
- Output type compatibility with downstream nodes

## Useful Resources

- **Official ComfyUI Docs**: https://docs.comfy.org/custom-nodes/walkthrough
- **Node Examples**: Check existing nodes in `ComfyUI/custom_nodes` for patterns
- **Testing**: Use ComfyUI's developer mode for detailed error messages