"""
Bayesian Optimization Nodes for ComfyUI
A comprehensive suite for optimizing diffusion model parameters using Bayesian methods
Works with image diffusion (SD, SDXL, Flux) and video diffusion models
"""

# Import unified system
from .unified_nodes import NODE_CLASS_MAPPINGS as UNIFIED_MAPPINGS
from .unified_nodes import NODE_DISPLAY_NAME_MAPPINGS as UNIFIED_DISPLAY

# Legacy nodes have been removed - functionality replaced by universal system
LEGACY_AVAILABLE = False

# Basic optimization nodes removed - functionality moved to flux_bayesian_nodes

# Import Flux-specific nodes
from .flux_bayesian_nodes import (
    EnhancedBayesianConfig,
    EnhancedParameterSampler,
    AestheticScorer,
    OptimizationDashboard,
    IterationCounter,
    ConditionalBranch,
    BayesianResultsExporter,
)

# Import adapter nodes if available
try:
    from .flux_adapter_nodes import (
        PowerLoraAdapter,
        ResolutionAdapter,
        SchedulerAdapter,
        SamplerAdapter,
        OptimizationLoopController,
        ParameterLogger,
        BatchParameterGenerator,
    )
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False

# Visualization nodes removed - functionality integrated into flux_bayesian_nodes
VIZ_AVAILABLE = False

# Start with unified mappings
NODE_CLASS_MAPPINGS = UNIFIED_MAPPINGS.copy()
NODE_DISPLAY_NAME_MAPPINGS = UNIFIED_DISPLAY.copy()

# Add legacy nodes if available
if LEGACY_AVAILABLE:
    legacy_mappings = {
        "BayesianParameterSpace": ParameterSpaceNode,
        "BayesianOptimizer": BayesianOptimizerNode,
        "BayesianMetricEvaluator": MetricEvaluatorNode,
        "BayesianSampler": BayesianSamplerNode,
        "BayesianHistoryLoader": BayesHistoryLoaderNode,
        "BayesianHistorySaver": BayesHistorySaverNode,
    }
    
    legacy_display = {
        "BayesianParameterSpace": "Parameter Space (Legacy)",
        "BayesianOptimizer": "Bayesian Optimizer (Legacy)",
        "BayesianMetricEvaluator": "Metric Evaluator (Legacy)",
        "BayesianSampler": "Bayesian Sampler (Legacy)",
        "BayesianHistoryLoader": "History Loader (Legacy)",
        "BayesianHistorySaver": "History Saver (Legacy)",
    }
    
    NODE_CLASS_MAPPINGS.update(legacy_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(legacy_display)

# Basic optimization nodes removed - using enhanced versions only

# Add Flux nodes
flux_mappings = {
    "EnhancedBayesianConfig": EnhancedBayesianConfig,
    "EnhancedParameterSampler": EnhancedParameterSampler,
    "AestheticScorer": AestheticScorer,
    "OptimizationDashboard": OptimizationDashboard,
    "IterationCounter": IterationCounter,
    "ConditionalBranch": ConditionalBranch,
    "BayesianResultsExporter": BayesianResultsExporter,
}

flux_display = {
    "EnhancedBayesianConfig": "Enhanced Config (Flux)",
    "EnhancedParameterSampler": "Enhanced Sampler (Flux)",
    "AestheticScorer": "Aesthetic Scorer",
    "OptimizationDashboard": "Optimization Dashboard",
    "IterationCounter": "Iteration Counter",
    "ConditionalBranch": "Conditional Branch",
    "BayesianResultsExporter": "Results Exporter",
}

NODE_CLASS_MAPPINGS.update(flux_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(flux_display)

# Add adapter nodes if available
if ADAPTERS_AVAILABLE:
    adapter_mappings = {
        "PowerLoraAdapter": PowerLoraAdapter,
        "ResolutionAdapter": ResolutionAdapter,
        "SchedulerAdapter": SchedulerAdapter,
        "SamplerAdapter": SamplerAdapter,
        "OptimizationLoopController": OptimizationLoopController,
        "ParameterLogger": ParameterLogger,
        "BatchParameterGenerator": BatchParameterGenerator,
    }
    
    adapter_display = {
        "PowerLoraAdapter": "Power LoRA Adapter",
        "ResolutionAdapter": "Resolution Adapter",
        "SchedulerAdapter": "Scheduler Adapter",
        "SamplerAdapter": "Sampler Adapter",
        "OptimizationLoopController": "Loop Controller",
        "ParameterLogger": "Parameter Logger",
        "BatchParameterGenerator": "Batch Generator",
    }
    
    NODE_CLASS_MAPPINGS.update(adapter_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(adapter_display)

# Visualization functionality is now integrated into flux_bayesian_nodes

# Version info
__version__ = "2.0.0"
__author__ = "Universal Bayesian Optimization for ComfyUI"

print(f"\nBayesian Optimization Nodes v{__version__} loaded successfully")
print(f"Total nodes available: {len(NODE_CLASS_MAPPINGS)}")
print(f"Universal system: ✓")
print(f"Legacy support: {'✓' if LEGACY_AVAILABLE else '✗'}")
print(f"Adapters: {'✓' if ADAPTERS_AVAILABLE else '✗'}")
print(f"Visualization: {'✓' if VIZ_AVAILABLE else '✗'}")

# Check for optional dependencies
try:
    import skopt
    print("\nscikit-optimize: ✓ (Full Bayesian optimization)")
except ImportError:
    print("\nscikit-optimize: ✗ (Install with: pip install scikit-optimize)")

try:
    import scipy
    print("scipy: ✓ (Advanced analysis)")
except ImportError:
    print("scipy: ✗ (Install with: pip install scipy)")

try:
    import lpips
    print("LPIPS: ✓ (Perceptual similarity)")
except ImportError:
    print("LPIPS: ✗ (Install with: pip install lpips)")

try:
    import clip
    print("CLIP: ✓ (Semantic similarity)")
except ImportError:
    print("CLIP: ✗ (Install with: pip install git+https://github.com/openai/CLIP.git)")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']