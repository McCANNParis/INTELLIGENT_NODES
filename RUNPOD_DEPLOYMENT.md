# Intelligent Nodes Deployment Guide for RunPod

## Quick Start for RunPod

This guide helps you deploy the Intelligent Self-Optimizing Nodes on RunPod's GPU cloud infrastructure.

## Prerequisites

- RunPod account with credits
- Basic familiarity with ComfyUI

## Step 1: Create RunPod Instance

1. Go to [RunPod.io](https://runpod.io)
2. Create a new GPU Pod with:
   - **Recommended GPU**: RTX 3090, RTX 4090, or A5000
   - **Template**: Select "ComfyUI" template (if available) or "PyTorch 2.0"
   - **Disk Space**: At least 50GB for models and optimization data
   - **Container Image** (if no ComfyUI template):
     ```
     runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel
     ```

## Step 2: Install ComfyUI (if not pre-installed)

SSH into your pod and run:

```bash
# Clone ComfyUI
cd /workspace
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Step 3: Install Intelligent Nodes

```bash
# Navigate to custom nodes directory
cd /workspace/ComfyUI/custom_nodes

# Clone this repository
git clone https://github.com/yourusername/NODES_BAYESIAN.git
cd NODES_BAYESIAN

# Install dependencies
pip install -r requirements.txt

# Install optional similarity metrics for better optimization
pip install optuna lpips dreamsim

# For CLIP support (optional but recommended)
pip install git+https://github.com/openai/CLIP.git
```

## Step 4: Download Models

```bash
# Create models directory for optimization data
mkdir -p /workspace/ComfyUI/models/optimizers

# Download Flux or other models (example)
cd /workspace/ComfyUI/models/checkpoints
wget https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors

# Download VAE if needed
cd /workspace/ComfyUI/models/vae
wget https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
```

## Step 5: Configure RunPod Networking

1. In RunPod dashboard, note your pod's Public IP and ports
2. ComfyUI typically runs on port 8188
3. Access URL will be: `http://YOUR_POD_IP:8188`

## Step 6: Start ComfyUI

```bash
cd /workspace/ComfyUI

# Start with increased memory for large models
python main.py --listen 0.0.0.0 --port 8188 --gpu-only --highvram
```

For RunPod's persistent storage:
```bash
# Start with specific paths for RunPod
python main.py --listen 0.0.0.0 --port 8188 \
  --output-directory /workspace/outputs \
  --input-directory /workspace/inputs \
  --gpu-only --highvram
```

## Step 7: Load Intelligent Workflow

1. Open ComfyUI in browser: `http://YOUR_POD_IP:8188`
2. Click "Load" in ComfyUI
3. Select `intelligent_workflow_example.json`
4. Upload your target image for optimization

## Step 8: Running Optimization

### Manual Mode (Interactive)
1. Click "Queue Prompt" to run one optimization trial
2. Each click improves the parameters
3. Watch the visualization update with progress
4. Best parameters are saved automatically

### Automated Mode (Batch)
Create a script for automated optimization:

```python
# /workspace/auto_optimize.py
import requests
import time
import json

COMFYUI_URL = "http://localhost:8188"
NUM_TRIALS = 100

# Load workflow
with open('/workspace/ComfyUI/custom_nodes/NODES_BAYESIAN/intelligent_workflow_example.json') as f:
    workflow = json.load(f)

# Run trials
for i in range(NUM_TRIALS):
    print(f"Running trial {i+1}/{NUM_TRIALS}")
    
    # Queue the prompt
    response = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow}
    )
    
    # Wait for completion
    time.sleep(30)  # Adjust based on generation time
    
    print(f"Trial {i+1} complete")

print("Optimization complete!")
```

Run with:
```bash
python /workspace/auto_optimize.py
```

## Step 9: Persistent Storage

RunPod pods can be terminated. To preserve optimization data:

### Option A: RunPod Network Volume
1. Create a Network Volume in RunPod dashboard
2. Mount it to `/workspace/persistent`
3. Configure nodes to save there:

```python
# In intelligent_nodes.py, modify get_studies_dir():
def get_studies_dir():
    # Use RunPod persistent storage
    persistent_dir = Path("/workspace/persistent/optimizers")
    persistent_dir.mkdir(parents=True, exist_ok=True)
    return persistent_dir
```

### Option B: Cloud Backup
```bash
# Backup to cloud storage (S3, GCS, etc.)
aws s3 sync /workspace/ComfyUI/models/optimizers s3://your-bucket/optimizers

# Restore on new pod
aws s3 sync s3://your-bucket/optimizers /workspace/ComfyUI/models/optimizers
```

## Step 10: Monitoring & Debugging

### View Logs
```bash
# ComfyUI logs
tail -f /workspace/comfyui.log

# Python logs
python -u main.py 2>&1 | tee comfyui.log
```

### Check GPU Usage
```bash
nvidia-smi -l 1  # Update every second
```

### Monitor Optimization Progress
The nodes create persistent study files in:
```
/workspace/ComfyUI/models/optimizers/
```

View study details:
```python
import pickle
import optuna

with open('/workspace/ComfyUI/models/optimizers/flux_optimization_*.pkl', 'rb') as f:
    study = pickle.load(f)

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
print(f"Number of trials: {len(study.trials)}")

# Visualize
optuna.visualization.plot_optimization_history(study).show()
```

## Performance Tips for RunPod

1. **Use High VRAM GPUs**: RTX 3090 (24GB) or better for Flux models
2. **Enable xformers**: Reduces memory usage
   ```bash
   pip install xformers
   python main.py --xformers
   ```
3. **Batch Processing**: Queue multiple prompts for efficiency
4. **Use SSD Storage**: RunPod NVMe is faster than network storage
5. **Optimize Batch Size**: Adjust based on GPU memory

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce memory usage
python main.py --lowvram --use-split-cross-attention
```

### Connection Issues
```bash
# Check if ComfyUI is running
curl http://localhost:8188/system_stats

# Check firewall
sudo ufw allow 8188
```

### Slow Generation
- Ensure you're using GPU: Check with `nvidia-smi`
- Reduce resolution in workflow
- Use fewer optimization steps initially

## Cost Optimization

1. **Spot Instances**: Use RunPod Spot for 50-80% savings
2. **Auto-pause**: Set pod to pause after inactivity
3. **Batch Runs**: Accumulate trials, run in batches
4. **Choose Right GPU**: 
   - Development: RTX 3060 (cheaper)
   - Production: RTX 3090/4090 (faster)

## Advanced Configuration

### Custom Samplers
Edit the parameter configuration in OptimizerSuggestNode:
```json
[
  {"name": "guidance", "type": "float", "low": 1.0, "high": 20.0},
  {"name": "steps", "type": "int", "low": 10, "high": 100},
  {"name": "sampler", "type": "categorical", 
   "choices": ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"]}
]
```

### Multi-GPU Setup
For multiple GPUs on RunPod:
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python main.py --listen 0.0.0.0

# Or distribute across GPUs (advanced)
python main.py --multi-gpu
```

## Integration with RunPod API

Automate pod management:
```python
import runpod

runpod.api_key = "your_api_key"

# Start pod
pod = runpod.create_pod(
    name="comfyui-optimizer",
    image_name="runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel",
    gpu_type_id="NVIDIA RTX 3090",
    cloud_type="SECURE",
    disk_size_in_gb=50
)

print(f"Pod started: {pod['id']}")
```

## Support

- **RunPod Discord**: For infrastructure issues
- **ComfyUI Discord**: For ComfyUI questions
- **GitHub Issues**: For node-specific problems

## Next Steps

1. Start with the example workflow
2. Adjust scoring weights for your use case
3. Run 20-50 trials to see optimization
4. Export best parameters for production
5. Create custom workflows for your specific needs

Happy optimizing! 🚀