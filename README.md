# Bayesian Optimization Nodes for ComfyUI

A comprehensive suite of custom nodes for optimizing Flux Dev parameters using Bayesian optimization methods. Automatically find the best CFG values, steps, schedulers, samplers, and LoRA weights for your specific use case.

## Features

- **Bayesian Optimization**: Efficiently explore parameter spaces using Gaussian Process regression
- **Flux-Specific Integration**: Optimized for Flux workflows with support for all major parameters
- **Multiple Similarity Metrics**: LPIPS, CLIP, MSE, SSIM, and aesthetic scoring
- **Real-time Visualization**: Monitor optimization progress with dashboards and convergence plots
- **Parameter Analysis**: Understand which parameters matter most for your use case
- **Export & Reporting**: Generate comprehensive reports in HTML, Markdown, or JSON

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/NODES_BAYESIAN.git
```

2. Install required dependencies:
```bash
pip install scikit-optimize torch torchvision matplotlib pillow scipy
```

3. Optional dependencies for advanced features:
```bash
pip install lpips clip scikit-learn seaborn
```

4. Restart ComfyUI

## Quick Start

### Basic Workflow

1. **Load Target Image**: Use the standard ComfyUI image loader
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

### Core Optimization Nodes

- **Enhanced Bayesian Config (Flux)**: Main configuration node for optimization parameters
- **Enhanced Parameter Sampler (Flux)**: Intelligent parameter sampling using Bayesian methods
- **Aesthetic Scorer**: Advanced similarity and quality scoring

### Adapter Nodes

- **Power LoRA Adapter**: Converts optimizer weights to Power Lora Loader format
- **Resolution Adapter**: Converts aspect ratios to width/height values
- **Scheduler Adapter**: Maps scheduler names to configurations
- **Sampler Adapter**: Maps sampler names to KSampler configurations
- **Optimization Loop Controller**: Controls the optimization flow

### Visualization Nodes

- **Optimization Dashboard**: Real-time monitoring of optimization progress
- **Parameter Heatmap**: Visualize parameter interactions
- **Convergence Plot**: Detailed convergence analysis
- **Parameter Importance Analysis**: Understand which parameters matter most

### Export Nodes

- **Optimization Report Generator**: Create comprehensive reports
- **Parameter Recommendation**: Get parameter suggestions based on results
- **Bayesian Results Exporter**: Export optimization history

## Parameter Ranges

### Recommended Starting Ranges

- **Guidance (CFG)**: 1.0 - 7.0
- **Steps**: 20 - 50
- **LoRA Weights**: 0.0 - 1.2
- **Schedulers**: beta, normal, simple, ddim_uniform
- **Samplers**: euler, uni_pc, dpmpp_2m, dpmpp_3m_sde

## Advanced Usage

### Multi-Stage Optimization

1. First optimize core parameters (CFG, steps)
2. Then fine-tune LoRA weights with fixed core parameters
3. Finally optimize aesthetic parameters

### Batch Processing

Use `Batch Parameter Generator` for parallel evaluation of multiple parameter sets.

### Custom Metrics

Implement your own scoring by modifying the `AestheticScorer` node or creating custom scorer nodes.

## Tips & Best Practices

1. **Start Small**: Begin with 30-50 iterations to test your setup
2. **Initial Exploration**: Use 10-15 initial points for good coverage
3. **Monitor Convergence**: Stop early if scores plateau
4. **Save Checkpoints**: Use the loop controller to save progress
5. **Analyze Results**: Use importance analysis to focus on key parameters

## Troubleshooting

### Slow Convergence
- Increase `n_initial_points` for better exploration
- Check if parameter ranges are too restrictive
- Ensure similarity metric matches your goal

### Memory Issues
- Reduce resolution during optimization
- Decrease batch size
- Use fewer LoRAs simultaneously

### Poor Results
- Verify target image quality
- Check if fixed prompt matches target
- Try different similarity metrics

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- ComfyUI community for the amazing framework
- scikit-optimize developers for Bayesian optimization tools
- Flux model creators for the powerful generation capabilities