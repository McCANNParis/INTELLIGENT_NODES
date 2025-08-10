# Automated Bayesian Optimization Workflow Guide

## Overview
This guide explains how to run fully automated Bayesian optimization with automatic image comparison in ComfyUI.

## Workflow Files

### 1. `bayesian-fully-automated.json` (RECOMMENDED)
**Features:**
- Fully automated with SimpleBayesianIterator node
- Automatic image saving and loading between iterations
- Real-time similarity calculation
- No manual intervention needed after initial setup

### 2. `bayesian-manual-iteration.json`
**Features:**
- Manual similarity score input
- Good for testing and debugging
- Requires manual similarity input for each iteration

### 3. `bayesian-auto-iteration.json`
**Features:**
- Uses AutoIterationLoader for automatic image loading
- Semi-automated approach
- Good balance between automation and control

## Setup Instructions

### Step 1: Load the Workflow
1. Open ComfyUI in your browser
2. Load `bayesian-fully-automated.json`
3. Verify all nodes are loaded correctly

### Step 2: Configure Initial Settings
1. **Target Image**: Upload your target image in the LoadImage node
2. **Prompt**: Update the prompt in CLIPTextEncode if needed
3. **Config**: Adjust parameters in EnhancedBayesianConfig:
   - Total iterations: 20 (default)
   - Initial samples: 5
   - Other optimization parameters as needed

### Step 3: Run First Iteration
1. Set the PrimitiveNode (is_first_run) to `true`
2. Click "Queue Prompt" to run iteration 1
3. Wait for completion

### Step 4: Setup Automation
1. Set the PrimitiveNode (is_first_run) to `false`
2. Open browser console (F12)
3. Copy and paste the contents of `run_automated_optimization.js`

### Step 5: Run Automated Optimization
In the browser console, run:
```javascript
runBayesianOptimization(19)  // Runs iterations 2-20
```

Or with custom delay:
```javascript
runBayesianOptimization(19, 15)  // 15 seconds between iterations
```

## How It Works

### SimpleBayesianIterator Node
1. **Saves** each generated image to `/workspace/ComfyUI/output/bayesian_iterations/`
2. **Calculates** MSE similarity between generated and target images
3. **Tracks** iteration count across runs
4. **Returns** similarity score to EnhancedParameterSampler

### EnhancedParameterSampler Node
1. **Receives** similarity score from SimpleBayesianIterator
2. **Updates** Bayesian optimization model with new data point
3. **Suggests** new parameters for next iteration
4. **Tracks** optimization progress and history

### Optimization Flow
```
Iteration 1 (Manual):
  Generate Image → Save → Calculate Similarity → Update Model

Iterations 2-20 (Automated):
  Load Previous → Generate New → Compare → Update → Repeat
```

## Output Structure
```
/workspace/ComfyUI/output/
├── bayesian_iterations/
│   ├── iteration_001.png
│   ├── iteration_002.png
│   ├── ...
│   └── iteration_020.png
├── bayesian_results_[timestamp].json
└── bayesian_optimization_state.pkl
```

## Monitoring Progress

### Browser Console
- Shows iteration progress
- Displays similarity scores
- Reports any errors

### ComfyUI Interface
- SaveImage node shows current iteration
- ShowText node displays iteration number
- OptimizationDashboard (if added) shows parameter trends

### Results Files
- `bayesian_results_[timestamp].json`: Final optimization results
- `bayesian_optimization_state.pkl`: Complete optimization history

## Troubleshooting

### Issue: "ImageSimilarityCalculator not found"
**Solution**: Ensure the node is registered in `__init__.py`

### Issue: "Maximum recursion depth exceeded"
**Solution**: Check for circular dependencies in workflow

### Issue: Iterations not improving
**Solutions**:
1. Check similarity calculation is working
2. Verify target image is loaded correctly
3. Adjust parameter ranges in config
4. Increase exploration vs exploitation

### Issue: Workflow stops after few iterations
**Solutions**:
1. Increase delay between iterations
2. Check browser console for errors
3. Ensure ComfyUI server is responsive

## Tips for Best Results

1. **Start Simple**: Test with 5 iterations first
2. **Monitor First Few**: Watch the first 2-3 iterations manually
3. **Adjust Delay**: Increase delay if your GPU is slower
4. **Check Similarity**: Verify similarity scores are meaningful (not all 0.0)
5. **Parameter Ranges**: Ensure parameter ranges are appropriate for your model

## Advanced Configuration

### Custom Similarity Metrics
Modify SimpleBayesianIterator to use different metrics:
- SSIM (Structural Similarity)
- LPIPS (Perceptual Similarity)
- CLIP (Semantic Similarity)

### Multi-Objective Optimization
Combine multiple metrics:
- Image similarity
- Aesthetic score
- Style consistency

### Batch Processing
Run multiple optimization sessions in parallel with different:
- Target images
- Prompts
- Parameter ranges

## Support

For issues or questions:
1. Check the browser console for errors
2. Review the optimization state file
3. Verify all nodes are properly connected
4. Ensure target image path is correct