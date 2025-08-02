"""
Bayesian Optimization Nodes for ComfyUI
A comprehensive suite for optimizing Flux parameters using Bayesian methods
"""

# Import all node classes
from .bayesian_optimization_nodes import (
    BayesianOptimizerConfig,
    BayesianParameterSampler,
    ImageSimilarityScorer,
    OptimizationVisualizer,
    BayesianResultsExporter,
)

from .flux_bayesian_nodes import (
    EnhancedBayesianConfig,
    EnhancedParameterSampler,
    AestheticScorer,
    OptimizationDashboard,
)

from .flux_adapter_nodes import (
    PowerLoraAdapter,
    ResolutionAdapter,
    SchedulerAdapter,
    SamplerAdapter,
    OptimizationLoopController,
    ParameterLogger,
    BatchParameterGenerator,
)

from .visualization_export_nodes import (
    ParameterHeatmap,
    ConvergencePlot,
    ParameterImportanceAnalysis,
    OptimizationReport,
    ParameterRecommendation,
)

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    # Basic Bayesian optimization nodes
    "BayesianOptimizerConfig": BayesianOptimizerConfig,
    "BayesianParameterSampler": BayesianParameterSampler,
    "ImageSimilarityScorer": ImageSimilarityScorer,
    "OptimizationVisualizer": OptimizationVisualizer,
    "BayesianResultsExporter": BayesianResultsExporter,
    
    # Enhanced Flux-specific nodes
    "EnhancedBayesianConfig": EnhancedBayesianConfig,
    "EnhancedParameterSampler": EnhancedParameterSampler,
    "AestheticScorer": AestheticScorer,
    "OptimizationDashboard": OptimizationDashboard,
    
    # Adapter nodes
    "PowerLoraAdapter": PowerLoraAdapter,
    "ResolutionAdapter": ResolutionAdapter,
    "SchedulerAdapter": SchedulerAdapter,
    "SamplerAdapter": SamplerAdapter,
    "OptimizationLoopController": OptimizationLoopController,
    "ParameterLogger": ParameterLogger,
    "BatchParameterGenerator": BatchParameterGenerator,
    
    # Visualization and export nodes
    "ParameterHeatmap": ParameterHeatmap,
    "ConvergencePlot": ConvergencePlot,
    "ParameterImportanceAnalysis": ParameterImportanceAnalysis,
    "OptimizationReport": OptimizationReport,
    "ParameterRecommendation": ParameterRecommendation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Basic nodes
    "BayesianOptimizerConfig": "Bayesian Optimizer Config",
    "BayesianParameterSampler": "Bayesian Parameter Sampler",
    "ImageSimilarityScorer": "Image Similarity Scorer",
    "OptimizationVisualizer": "Optimization Visualizer",
    "BayesianResultsExporter": "Bayesian Results Exporter",
    
    # Enhanced nodes
    "EnhancedBayesianConfig": "Enhanced Bayesian Config (Flux)",
    "EnhancedParameterSampler": "Enhanced Parameter Sampler (Flux)",
    "AestheticScorer": "Aesthetic Scorer",
    "OptimizationDashboard": "Optimization Dashboard",
    
    # Adapter nodes
    "PowerLoraAdapter": "Power LoRA Adapter",
    "ResolutionAdapter": "Resolution Adapter",
    "SchedulerAdapter": "Scheduler Adapter",
    "SamplerAdapter": "Sampler Adapter",
    "OptimizationLoopController": "Optimization Loop Controller",
    "ParameterLogger": "Parameter Logger",
    "BatchParameterGenerator": "Batch Parameter Generator",
    
    # Visualization nodes
    "ParameterHeatmap": "Parameter Heatmap",
    "ConvergencePlot": "Convergence Plot",
    "ParameterImportanceAnalysis": "Parameter Importance Analysis",
    "OptimizationReport": "Optimization Report Generator",
    "ParameterRecommendation": "Parameter Recommendation",
}

# Version info
__version__ = "1.0.0"
__author__ = "Bayesian Optimization for ComfyUI"

print(f"Bayesian Optimization Nodes v{__version__} loaded successfully")
print(f"Total nodes available: {len(NODE_CLASS_MAPPINGS)}")

# Optional: Check for required dependencies
try:
    import skopt
    print("scikit-optimize available - Full optimization features enabled")
except ImportError:
    print("Warning: scikit-optimize not installed - Using basic optimization")
    print("Install with: pip install scikit-optimize")

try:
    import scipy
    print("scipy available - Advanced analysis features enabled")
except ImportError:
    print("Warning: scipy not installed - Some analysis features disabled")
    print("Install with: pip install scipy")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']