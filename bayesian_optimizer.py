import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.stats import norm
import json
from typing import Dict, List, Tuple, Any, Union


class BayesianOptimizer:
    def __init__(self, parameter_space: Dict[str, Any]):
        self.param_space = parameter_space
        self.gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
        
    def encode_parameters(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Encode parameters to numerical array for GP"""
        encoded = []
        
        for key, value in sorted(parameters.items()):
            if key not in self.param_space:
                continue
                
            param_def = self.param_space[key]
            
            if isinstance(param_def, tuple):  # Continuous parameter
                min_val, max_val = param_def
                normalized = (value - min_val) / (max_val - min_val)
                encoded.append(normalized)
            elif isinstance(param_def, list):  # Categorical parameter
                if isinstance(value, (int, float)):
                    encoded.append(value / len(param_def))
                else:
                    idx = param_def.index(value) if value in param_def else 0
                    encoded.append(idx / len(param_def))
        
        return np.array(encoded)
    
    def decode_parameters(self, encoded: np.ndarray) -> Dict[str, Any]:
        """Decode numerical array back to parameters"""
        parameters = {}
        idx = 0
        
        for key in sorted(self.param_space.keys()):
            param_def = self.param_space[key]
            
            if isinstance(param_def, tuple):  # Continuous parameter
                min_val, max_val = param_def
                value = encoded[idx] * (max_val - min_val) + min_val
                if isinstance(min_val, int) and isinstance(max_val, int):
                    value = int(round(value))
                parameters[key] = value
            elif isinstance(param_def, list):  # Categorical parameter
                cat_idx = int(encoded[idx] * len(param_def))
                cat_idx = min(cat_idx, len(param_def) - 1)
                parameters[key] = param_def[cat_idx]
            
            idx += 1
        
        return parameters
    
    def sample_random(self) -> Dict[str, Any]:
        """Sample random parameters from the space"""
        parameters = {}
        
        for key, param_def in self.param_space.items():
            if isinstance(param_def, tuple):  # Continuous parameter
                min_val, max_val = param_def
                if isinstance(min_val, int) and isinstance(max_val, int):
                    value = np.random.randint(min_val, max_val + 1)
                else:
                    value = np.random.uniform(min_val, max_val)
                parameters[key] = value
            elif isinstance(param_def, list):  # Categorical parameter
                parameters[key] = np.random.choice(param_def)
        
        return parameters
    
    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate Expected Improvement acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        with np.errstate(divide='warn'):
            Z = (mu - self.best_score - xi) / sigma
            ei = (mu - self.best_score - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei.flatten()
    
    def upper_confidence_bound(self, X: np.ndarray, kappa: float = 2.0) -> np.ndarray:
        """Calculate Upper Confidence Bound acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + kappa * sigma
    
    def probability_of_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate Probability of Improvement acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        Z = (mu - self.best_score - xi) / sigma
        return norm.cdf(Z)
    
    def optimize_acquisition(self, acquisition: str = 'EI') -> np.ndarray:
        """Optimize the acquisition function"""
        dim = len(self.param_space)
        
        # Define acquisition function
        if acquisition == 'EI':
            acq_func = lambda x: -self.expected_improvement(x.reshape(1, -1))
        elif acquisition == 'UCB':
            acq_func = lambda x: -self.upper_confidence_bound(x.reshape(1, -1))
        elif acquisition == 'PI':
            acq_func = lambda x: -self.probability_of_improvement(x.reshape(1, -1))
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}")
        
        # Multi-start optimization
        best_x = None
        best_acq = np.inf
        
        for _ in range(10):
            x0 = np.random.rand(dim)
            result = minimize(
                acq_func,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * dim
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        return best_x
    
    def suggest_next(self, acquisition: str = 'EI') -> Dict[str, Any]:
        """Suggest next parameters to evaluate"""
        if len(self.X_observed) < 3:
            # Random exploration for first few points
            return self.sample_random()
        
        # Fit GP on observed data
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)
        
        # Optimize acquisition function
        best_x = self.optimize_acquisition(acquisition)
        
        return self.decode_parameters(best_x)
    
    def update(self, parameters: Dict[str, Any], score: float):
        """Update optimizer with new observation"""
        encoded_params = self.encode_parameters(parameters)
        self.X_observed.append(encoded_params)
        self.y_observed.append(score)
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = parameters.copy()
    
    def get_history(self) -> Dict[str, Any]:
        """Get optimization history"""
        return {
            'X_observed': [x.tolist() for x in self.X_observed],
            'y_observed': self.y_observed,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'param_space': self.param_space
        }
    
    def load_history(self, history: Dict[str, Any]):
        """Load optimization history"""
        self.X_observed = [np.array(x) for x in history['X_observed']]
        self.y_observed = history['y_observed']
        self.best_params = history.get('best_params')
        self.best_score = history.get('best_score', -np.inf)
        if 'param_space' in history:
            self.param_space = history['param_space']