from .nodes import (
    BayesianOptimizerNode, 
    ParameterSpaceNode, 
    MetricEvaluatorNode, 
    BayesianSamplerNode,
    BayesHistoryLoaderNode,
    BayesHistorySaverNode
)

NODE_CLASS_MAPPINGS = {
    "BayesianOptimizer": BayesianOptimizerNode,
    "ParameterSpace": ParameterSpaceNode,
    "MetricEvaluator": MetricEvaluatorNode,
    "BayesianSampler": BayesianSamplerNode,
    "BayesHistoryLoader": BayesHistoryLoaderNode,
    "BayesHistorySaver": BayesHistorySaverNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BayesianOptimizer": "Bayesian Optimizer",
    "ParameterSpace": "Parameter Space",
    "MetricEvaluator": "Metric Evaluator",
    "BayesianSampler": "Bayesian Sampler",
    "BayesHistoryLoader": "Load Optimization History",
    "BayesHistorySaver": "Save Optimization History",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']