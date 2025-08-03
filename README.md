# Bayesian Optimization Nodes for ComfyUI

A comprehensive suite of custom nodes for optimizing diffusion model parameters using Bayesian optimization methods. This universal system supports image models (Stable Diffusion, SDXL, Flux) and video models, automatically finding the best parameters for your specific use case.

## Key Features

### Universal Model Support
- **Auto-detection**: Automatically detects model types (image/video)
- **Multiple Models**: Works with SD, SDXL, Flux, AnimateDiff, ModelScope, etc.
- **Unified Interface**: Consistent node interface across all model types
- **Smart Adapters**: Model-specific optimizations through adapters

### Core Optimization Features
- **Bayesian Optimization**: Efficiently explore parameter spaces using Gaussian Process regression
- **Multiple Metrics**: LPIPS, CLIP, MSE, SSIM, aesthetic scoring, and temporal consistency
- **Real-time Visualization**: Monitor optimization progress with dashboards
- **Parameter Analysis**: Understand which parameters matter most
- **Export & Reporting**: Generate comprehensive reports in HTML, Markdown, or JSON

### Workflow Options
- **AutoModelOptimizer**: One-click optimization for any model
- **Simple Workflows**: Easy setup for beginners with presets
- **Advanced Workflows**: Full control for power users
- **Batch Processing**: Optimize multiple targets simultaneously

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/NODES_BAYESIAN.git
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. For advanced features, install optional dependencies:
```bash
# For perceptual similarity metrics
pip install lpips

# For semantic similarity
pip install git+https://github.com/openai/CLIP.git
```

4. Restart ComfyUI

## Quick Start

### Simplest Workflow (Auto-Optimization)
1. Connect your model → `AutoModelOptimizer`
2. Provide target image/video and prompt
3. Set optimization steps
4. Run!

### Basic Flux Workflow
1. **Load Target Image**: Use standard ComfyUI image loader
2. **Configure Optimization**: Add `Enhanced Bayesian Config (Flux)` node
3. **Sample Parameters**: Connect `Enhanced Parameter Sampler (Flux)` 
4. **Adapt Parameters**: Use adapter nodes to convert optimizer outputs
5. **Score Results**: Add `Aesthetic Scorer` to evaluate generated images
6. **Monitor Progress**: Use `Optimization Dashboard` for real-time visualization

### Example Connection Flow
```
[Load Image] → [Enhanced Bayesian Config]
                        ↓
              [Enhanced Parameter Sampler]
                        ↓
    ┌────────────────────────────────────────┐
    ↓              ↓              ↓           ↓
[guidance]    [steps]     [scheduler]   [sampler]
    ↓              ↓              ↓           ↓
[FluxGuidance] [Scheduler]  [Scheduler]  [KSampler]
                                          ↓
                                    [Generated Image]
                                          ↓
                                  [Aesthetic Scorer]
                                          ↓
                              [Back to Parameter Sampler]
```

## Node Categories

### Universal System Nodes
- **UniversalParameterSpace**: Define parameters for any model
- **UniversalBayesianOptimizer**: Core optimization engine
- **UniversalSampler**: Sample with any diffusion model
- **MetricEvaluatorUniversal**: Evaluate any content type

### Auto/Simple Nodes
- **AutoModelOptimizer**: Automatic optimization
- **SimpleOptimizationSetup**: Easy setup interface
- **SimpleOptimizationRun**: Run optimization
- **PresetOptimizationConfigs**: Pre-configured settings

### Flux-Specific Nodes
- **Enhanced Bayesian Config (Flux)**: Configuration for Flux optimization
- **Enhanced Parameter Sampler (Flux)**: Intelligent parameter sampling
- **Aesthetic Scorer**: Advanced similarity and quality scoring
- **Optimization Dashboard**: Real-time monitoring

### Adapter Nodes
- **Power LoRA Adapter**: Converts optimizer weights to Power Lora format
- **Resolution Adapter**: Converts aspect ratios to width/height
- **Scheduler Adapter**: Maps scheduler names to configurations
- **Sampler Adapter**: Maps sampler names to KSampler configs
- **Optimization Loop Controller**: Controls optimization flow
- **Parameter Logger**: Logs optimization progress
- **Batch Parameter Generator**: Generate multiple parameter sets

## Parameter Ranges

### Recommended Starting Ranges
- **Guidance (CFG)**: 1.0 - 7.0
- **Steps**: 20 - 50
- **LoRA Weights**: 0.0 - 1.2
- **Schedulers**: beta, normal, simple, ddim_uniform, karras, exponential
- **Samplers**: euler, uni_pc, dpmpp_2m, dpmpp_3m_sde, heun, dpm_2
- **Resolution Ratios**: 1:1, 3:4, 4:3, 16:9, 9:16

## Advanced Usage

### Multi-Stage Optimization
1. First optimize core parameters (CFG, steps)
2. Then fine-tune LoRA weights with fixed core parameters
3. Finally optimize aesthetic parameters

### Video Optimization
1. Use `UniversalParameterSpace` with model_type="video"
2. Configure temporal parameters
3. Use `MetricEvaluatorUniversal` with temporal metrics
4. Optimize with temporal consistency metrics

### Custom Workflows
1. Define custom parameter space with `UniversalParameterSpace`
2. Add custom parameters via JSON
3. Use appropriate metric evaluator
4. Build complex optimization pipelines

### Batch Processing
Use `Batch Parameter Generator` with strategies:
- **Sobol**: Quasi-random sequence for good coverage
- **Latin Hypercube**: Optimal space-filling design
- **Grid**: Systematic parameter exploration
- **Random**: Pure random sampling

## Performance Optimizations
- Reduced code duplication through universal system
- Efficient parameter encoding/decoding
- Batched operations where possible
- Smart caching of model evaluations
- Minimal memory footprint

## Tips & Best Practices

1. **Start Small**: Begin with 30-50 iterations to test your setup
2. **Initial Exploration**: Use 10-15 initial points for good coverage
3. **Monitor Convergence**: Stop early if scores plateau
4. **Save Checkpoints**: Use loop controller to save progress
5. **Analyze Results**: Use importance analysis to focus on key parameters
6. **Use Presets**: Start with preset configurations for common use cases

## Troubleshooting

### Slow Convergence
- Increase `n_initial_points` for better exploration
- Check if parameter ranges are too restrictive
- Ensure similarity metric matches your goal

### Memory Issues
- Reduce resolution during optimization
- Decrease batch size
- Use fewer LoRAs simultaneously
- Enable gradient checkpointing if available

### Poor Results
- Verify target image/video quality
- Check if fixed prompt matches target
- Try different similarity metrics
- Adjust metric weights for your use case

## File Structure
```
NODES_BAYESIAN/
├── core/                   # Core optimization system
│   ├── base_optimizer.py   # Base classes
│   └── __init__.py
├── adapters/              # Model adapters
│   ├── image_adapter.py   # Image models
│   ├── video_adapter.py   # Video models
│   └── __init__.py
├── unified_nodes.py       # Main unified node system
├── metrics_universal.py   # Universal metrics
├── flux_bayesian_nodes.py # Flux-specific nodes
├── flux_adapter_nodes.py  # Adapter nodes
└── __init__.py           # Main entry point
```

## Migration Guide

### From Legacy Nodes
Replace:
- `ParameterSpaceNode` → `UniversalParameterSpace`
- `BayesianOptimizerNode` → `UniversalBayesianOptimizer`
- `BayesianSamplerNode` → `UniversalSampler`
- `MetricEvaluatorNode` → `MetricEvaluatorUniversal`

### From Basic/Original Nodes
- Most nodes have universal equivalents
- Use `AutoModelOptimizer` for quick setup
- Enhanced nodes provide superset of functionality

## Future Enhancements
- Audio diffusion model support
- Multi-GPU optimization
- Distributed optimization
- Real-time parameter adjustment
- Cloud-based optimization services
- Integration with more model types

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- ComfyUI community for the amazing framework
- scikit-optimize developers for Bayesian optimization tools
- Model creators for the powerful generation capabilities