# Testing Guide for Intelligent Nodes

## Quick Test Workflows

This package includes several test workflows to verify the system works correctly:

### 1. Minimal Test (`test_workflow_minimal.json`)
**Purpose**: Test the optimization loop in isolation without image generation

**What it does**:
- Creates a simple optimization that maximizes a parameter from 0 to 1
- Uses the parameter value directly as the score (self-feedback)
- Should converge to 1.0 after several trials

**How to test**:
1. Load `test_workflow_minimal.json`
2. Click "Queue Prompt" 5-10 times
3. Watch the trial count increase
4. Observe the best value approach 1.0
5. Check that status messages update correctly

**Expected results**:
- Trial count increases with each run
- Best value improves towards 1.0
- No errors in console

### 2. Simple Test (`test_workflow_simple.json`)
**Purpose**: Test full image generation and scoring pipeline

**What it does**:
- Optimizes cfg, steps, and sampler for SDXL
- Compares generated images to a target image
- Uses MSE scoring (simple pixel comparison)

**How to test**:
1. Load `test_workflow_simple.json`
2. Load a target image (512x512 recommended)
3. Make sure you have SDXL checkpoint loaded
4. Click "Queue Prompt" repeatedly
5. Watch optimization progress

**Expected results**:
- Images generate successfully
- Scores are calculated and displayed
- Optimization improves over trials
- Visualization updates (if matplotlib installed)

### 3. Full Example (`intelligent_workflow_example.json`)
**Purpose**: Complete demonstration with all features

**Includes**:
- SamplerAdapter for type conversion
- Multiple scoring metrics (if installed)
- Visualization node
- Full parameter space optimization

## Troubleshooting

### Common Issues and Solutions

#### 1. "No trials are completed yet" Error
**Solution**: This is normal on first run. The error has been fixed in the latest version.

#### 2. Type Mismatch for Sampler
**Solution**: Use the SamplerAdapter node between OptimizerSuggestNode and KSampler.

#### 3. Missing Dependencies
```bash
# Install required packages
pip install optuna torch torchvision numpy pillow matplotlib

# Optional for advanced scoring
pip install lpips dreamsim
```

#### 4. Workflow Validation Errors
- Make sure all links are properly connected
- Check that node IDs match link references
- Use the minimal test workflow to isolate issues

## Testing Checklist

### Basic Functionality
- [ ] OptimizerStateNode creates/loads study
- [ ] OptimizerSuggestNode proposes parameters
- [ ] Parameters flow to generation nodes
- [ ] OptimizerTellNode completes trials
- [ ] Study persists between runs

### Type Compatibility
- [ ] FLOAT parameters work with KSampler cfg
- [ ] INT parameters work with KSampler steps
- [ ] STRING → SamplerAdapter → COMBO works
- [ ] STUDY and TRIAL objects pass correctly

### Persistence
- [ ] Studies save to disk
- [ ] Studies reload after ComfyUI restart
- [ ] Best parameters are preserved
- [ ] Trial history is maintained

### Scoring
- [ ] MSE scoring works
- [ ] Multiple metrics combine correctly (if available)
- [ ] Scores feed back to optimizer
- [ ] Best score updates properly

## Manual Testing Steps

### Step 1: Verify Installation
```python
# In Python console
import optuna
import torch
print("Optuna version:", optuna.__version__)
print("PyTorch version:", torch.__version__)
```

### Step 2: Test Minimal Loop
1. Load `test_workflow_minimal.json`
2. Queue 5 times
3. Verify trial count = 5
4. Verify best value > 0.8

### Step 3: Test with Images
1. Load `test_workflow_simple.json`
2. Set a simple target (solid color works well)
3. Queue 10 times
4. Verify images generate
5. Verify scores calculate

### Step 4: Test Persistence
1. Run 5 trials
2. Restart ComfyUI
3. Load same workflow
4. Verify trial count preserved
5. Continue optimization

## Performance Expectations

- **Trial Speed**: 1-30 seconds per trial (depends on model/resolution)
- **Convergence**: 10-50 trials for simple targets
- **Memory Usage**: ~2-4GB for optimization state
- **Storage**: ~1MB per 100 trials

## Debug Mode

To enable verbose logging, edit `intelligent_nodes.py`:

```python
# Add at top of file
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show:
- Study creation/loading
- Parameter suggestions
- Score calculations
- Trial completions

## Reporting Issues

When reporting issues, please include:
1. The workflow JSON you're using
2. ComfyUI console output
3. Python version and OS
4. Installed packages (`pip list`)
5. Steps to reproduce

## Next Steps

Once basic tests pass:
1. Try optimizing for your own images
2. Adjust parameter ranges in OptimizerSuggestNode
3. Experiment with different scoring weights
4. Create custom workflows for your use case

Happy optimizing! 🚀