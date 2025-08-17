# 🧠 Intelligent Self-Optimizing Nodes for ComfyUI

Transform your ComfyUI workflows into self-improving machines that learn and optimize with each generation.

## 🌟 What Makes This Different?

Unlike traditional workflows where you manually tweak parameters, these intelligent nodes create a **self-optimizing system** that:
- **Learns** from each generation
- **Remembers** what works best (persistent storage)
- **Suggests** better parameters automatically
- **Visualizes** optimization progress in real-time

Each click of "Queue Prompt" isn't just generating an image - it's teaching the system to be better.

## 🚀 Quick Start

### Installation

1. Navigate to ComfyUI custom nodes:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/NODES_BAYESIAN.git
cd NODES_BAYESIAN
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install advanced similarity metrics:
```bash
pip install lpips dreamsim
```

5. Restart ComfyUI

### First Optimization

1. Load `intelligent_workflow_example.json` in ComfyUI
2. Add your target image (what you want to match)
3. Click "Queue Prompt" to start learning
4. Watch as each generation gets better!

## 📦 Node Overview

### Core Optimization Nodes

#### 🧠 Optimizer State
The "brain" that manages the optimization study
- Creates and loads persistent optimization studies
- Tracks all trials and their results
- Survives ComfyUI restarts

#### 💡 Suggest Parameters  
Proposes the next set of parameters to test
- Configurable parameter spaces (float, int, categorical)
- Uses advanced optimization algorithms (TPE, CMA-ES)
- Can replay best parameters

#### 📊 Score Images
Evaluates how well generated images match your target
- Multiple metrics: DreamSim, LPIPS, MSE
- Weighted combination for custom objectives
- Real-time scoring feedback

#### ✍️ Complete Trial
Closes the optimization loop
- Reports results back to the optimizer
- Updates the internal model
- Triggers auto-save

### Utility Nodes

#### 🏆 Best Parameters
Extracts the optimal configuration found so far

#### 📈 Visualize Study
Creates plots showing optimization progress

#### 🔄 Pass Trial
Helper for complex workflow routing

## 🎯 Use Cases

- **Style Transfer**: Find optimal parameters to match a reference style
- **Parameter Tuning**: Automatically discover the best CFG, steps, sampler
- **Model Comparison**: Optimize across different checkpoints
- **Prompt Engineering**: Test prompt variations systematically
- **Quality Optimization**: Maximize aesthetic scores

## 🔧 Advanced Configuration

### Custom Parameter Spaces

Configure the OptimizerSuggestNode with JSON:
```json
[
  {"name": "cfg", "type": "float", "low": 1.0, "high": 20.0},
  {"name": "steps", "type": "int", "low": 20, "high": 100},
  {"name": "sampler", "type": "categorical", 
   "choices": ["euler", "dpmpp_2m", "ddim"]}
]
```

### Optimization Algorithms

- **TPE** (Default): Tree-structured Parzen Estimator - great for most cases
- **Random**: Baseline for comparison
- **CMA-ES**: Evolution strategy - good for continuous parameters

### Scoring Weights

Adjust the importance of different similarity metrics:
- **DreamSim**: Human perceptual similarity (0.0-1.0)
- **LPIPS**: Learned perceptual similarity (0.0-1.0)
- **MSE**: Pixel-level similarity (0.0-1.0)

## 💾 Persistence

Studies are automatically saved to:
```
ComfyUI/models/optimizers/[study_name]_[hash].pkl
```

This means:
- ✅ Optimization continues after restarts
- ✅ Studies can be shared between machines
- ✅ Progress is never lost

## 🚅 RunPod Deployment

See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for detailed cloud deployment instructions.

Quick setup:
```bash
# On RunPod GPU instance
cd /workspace/ComfyUI/custom_nodes
git clone [this-repo]
cd NODES_BAYESIAN
pip install -r requirements.txt
```

## 📊 Monitoring Progress

Watch optimization in real-time:
1. Use the **StudyVisualizerNode** to see optimization history
2. Check the **best_value** output to track improvements
3. Monitor the **status** output for trial information

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional similarity metrics
- New optimization algorithms
- UI improvements
- Workflow templates

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- [Optuna](https://optuna.org/) - Advanced optimization framework
- [DreamSim](https://dreamsim-nights.github.io/) - Human perceptual similarity
- [LPIPS](https://richzhang.github.io/PerceptualSimilarity/) - Learned perceptual metrics
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The amazing node-based UI

## 💬 Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Discord**: ComfyUI Discord #custom-nodes channel

---

**Remember**: Every click makes it smarter! 🚀