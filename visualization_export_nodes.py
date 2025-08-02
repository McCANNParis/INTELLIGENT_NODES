"""
Advanced visualization and export nodes for Bayesian optimization results
"""

import numpy as np
import torch
import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import io

# Try to import optional dependencies
try:
    from scipy import stats
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class ParameterHeatmap:
    """Creates a heatmap visualization of parameter interactions"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
                "param_x": (["guidance", "steps", "lora1_weight", "lora2_weight"],),
                "param_y": (["guidance", "steps", "lora1_weight", "lora2_weight"],),
            },
            "optional": {
                "resolution": ("INT", {"default": 50, "min": 20, "max": 200}),
                "cmap": (["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_heatmap"
    CATEGORY = "Bayesian Optimization/Visualization"
    
    def create_heatmap(self, config, param_x, param_y, resolution=50, cmap="viridis"):
        if not config["history"] or len(config["history"]) < 5:
            # Return placeholder
            placeholder = np.ones((512, 512, 3), dtype=np.uint8) * 255
            return (torch.from_numpy(placeholder).float() / 255.0,)
        
        # Extract data
        x_values = []
        y_values = []
        scores = []
        
        for entry in config["history"]:
            if param_x in entry["params"] and param_y in entry["params"]:
                x_values.append(entry["params"][param_x])
                y_values.append(entry["params"][param_y])
                scores.append(entry["score"])
        
        if not x_values:
            placeholder = np.ones((512, 512, 3), dtype=np.uint8) * 255
            return (torch.from_numpy(placeholder).float() / 255.0,)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if SCIPY_AVAILABLE and len(x_values) > 10:
            # Create interpolated heatmap
            xi = np.linspace(min(x_values), max(x_values), resolution)
            yi = np.linspace(min(y_values), max(y_values), resolution)
            xi, yi = np.meshgrid(xi, yi)
            
            # Interpolate scores
            zi = griddata((x_values, y_values), scores, (xi, yi), method='cubic')
            
            # Plot heatmap
            im = ax.imshow(zi, extent=[min(x_values), max(x_values), 
                                      min(y_values), max(y_values)],
                          origin='lower', cmap=cmap, aspect='auto')
            
            # Add scatter points
            scatter = ax.scatter(x_values, y_values, c=scores, cmap=cmap, 
                               edgecolors='black', linewidth=1, s=50, alpha=0.8)
        else:
            # Simple scatter plot with color coding
            scatter = ax.scatter(x_values, y_values, c=scores, cmap=cmap, 
                               s=100, edgecolors='black', linewidth=1)
            im = scatter
        
        # Mark best point
        if config["best_params"]:
            if param_x in config["best_params"] and param_y in config["best_params"]:
                ax.scatter(config["best_params"][param_x], 
                         config["best_params"][param_y],
                         marker='*', s=500, c='red', edgecolors='white', 
                         linewidth=2, label=f'Best: {config["best_score"]:.3f}')
        
        # Labels and title
        ax.set_xlabel(param_x.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(param_y.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Parameter Interaction: {param_x} vs {param_y}', fontsize=14)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity Score', rotation=270, labelpad=20)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Legend if best point exists
        if config["best_params"]:
            ax.legend()
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close()
        
        return (torch.from_numpy(img_array).float() / 255.0,)

class ConvergencePlot:
    """Creates detailed convergence analysis plots"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
            },
            "optional": {
                "show_confidence": ("BOOLEAN", {"default": True}),
                "show_regret": ("BOOLEAN", {"default": True}),
                "show_diversity": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_convergence_plot"
    CATEGORY = "Bayesian Optimization/Visualization"
    
    def create_convergence_plot(self, config, show_confidence=True, 
                               show_regret=True, show_diversity=True):
        if not config["history"]:
            placeholder = np.ones((512, 512, 3), dtype=np.uint8) * 255
            return (torch.from_numpy(placeholder).float() / 255.0,)
        
        # Determine number of subplots
        n_plots = 1 + sum([show_confidence, show_regret, show_diversity])
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # Extract data
        iterations = [h["iteration"] for h in config["history"]]
        scores = [h["score"] for h in config["history"]]
        
        # Plot 1: Main convergence with running best
        ax_idx = 0
        ax = axes[ax_idx]
        
        # Plot all scores
        ax.plot(iterations, scores, 'o-', alpha=0.5, label='Score', color='blue')
        
        # Plot running best
        running_best = []
        current_best = float('-inf')
        for score in scores:
            current_best = max(current_best, score)
            running_best.append(current_best)
        
        ax.plot(iterations, running_best, '-', linewidth=2, 
               label='Best so far', color='red')
        
        # Mark exploration vs exploitation phases
        if config["n_initial_points"] < len(iterations):
            ax.axvline(x=config["n_initial_points"], color='gray', 
                      linestyle='--', alpha=0.5, label='Exploration → Exploitation')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.set_title('Optimization Convergence')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Confidence intervals
        if show_confidence and SCIPY_AVAILABLE and len(scores) > 10:
            ax_idx += 1
            ax = axes[ax_idx]
            
            # Calculate rolling statistics
            window = min(10, len(scores) // 3)
            rolling_mean = []
            rolling_std = []
            
            for i in range(len(scores)):
                start = max(0, i - window // 2)
                end = min(len(scores), i + window // 2 + 1)
                window_scores = scores[start:end]
                rolling_mean.append(np.mean(window_scores))
                rolling_std.append(np.std(window_scores))
            
            rolling_mean = np.array(rolling_mean)
            rolling_std = np.array(rolling_std)
            
            # Plot with confidence bands
            ax.plot(iterations, rolling_mean, '-', linewidth=2, 
                   label='Rolling mean', color='green')
            ax.fill_between(iterations, 
                           rolling_mean - 2 * rolling_std,
                           rolling_mean + 2 * rolling_std,
                           alpha=0.3, color='green',
                           label='95% confidence')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Score')
            ax.set_title('Score Stability Analysis')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 3: Simple regret
        if show_regret:
            ax_idx += 1
            ax = axes[ax_idx]
            
            # Calculate simple regret (distance from best found)
            if config["best_score"] > 0:
                regret = [config["best_score"] - score for score in scores]
                ax.semilogy(iterations, regret, 'o-', color='orange')
                ax.set_ylabel('Simple Regret (log scale)')
            else:
                # If no positive scores, show improvement needed
                improvement_needed = [max(scores) - score for score in scores]
                ax.plot(iterations, improvement_needed, 'o-', color='orange')
                ax.set_ylabel('Improvement Needed')
            
            ax.set_xlabel('Iteration')
            ax.set_title('Optimization Regret')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Parameter diversity
        if show_diversity:
            ax_idx += 1
            ax = axes[ax_idx]
            
            # Calculate parameter diversity over time
            diversity_scores = []
            
            for i in range(1, len(config["history"]) + 1):
                # Get last N iterations
                window_size = min(10, i)
                recent_history = config["history"][max(0, i - window_size):i]
                
                # Calculate diversity as variance in parameters
                param_variances = []
                
                # Check continuous parameters
                for param in ['guidance', 'steps']:
                    if param in recent_history[0]["params"]:
                        values = [h["params"][param] for h in recent_history]
                        if len(values) > 1:
                            param_variances.append(np.var(values))
                
                diversity = np.mean(param_variances) if param_variances else 0
                diversity_scores.append(diversity)
            
            ax.plot(iterations, diversity_scores, '-', linewidth=2, color='purple')
            ax.fill_between(iterations, 0, diversity_scores, alpha=0.3, color='purple')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Parameter Diversity')
            ax.set_title('Exploration Diversity Over Time')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close()
        
        return (torch.from_numpy(img_array).float() / 255.0,)

class ParameterImportanceAnalysis:
    """Analyzes and visualizes parameter importance"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
            },
            "optional": {
                "method": (["correlation", "mutual_information", "variance", "shap"],),
                "top_k": ("INT", {"default": 10, "min": 3, "max": 20}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("importance_plot", "importance_summary")
    FUNCTION = "analyze_importance"
    CATEGORY = "Bayesian Optimization/Visualization"
    
    def analyze_importance(self, config, method="correlation", top_k=10):
        if not config["history"] or len(config["history"]) < 10:
            placeholder = np.ones((512, 512, 3), dtype=np.uint8) * 255
            return (torch.from_numpy(placeholder).float() / 255.0, 
                   "Not enough data for importance analysis")
        
        # Extract parameter values and scores
        param_data = {}
        scores = []
        
        # Collect all parameters
        for entry in config["history"]:
            scores.append(entry["score"])
            
            for param, value in entry["params"].items():
                if param not in param_data:
                    param_data[param] = []
                
                # Convert categorical to numeric
                if isinstance(value, str):
                    # For categorical variables, use one-hot encoding
                    if param == "scheduler":
                        value = config["param_names"]["schedulers"].index(value)
                    elif param == "sampler":
                        value = config["param_names"]["samplers"].index(value)
                    elif param == "resolution_ratio":
                        value = config["param_names"]["ratios"].index(value)
                
                param_data[param].append(float(value) if value is not None else 0.0)
        
        # Calculate importance scores
        importance_scores = {}
        
        if method == "correlation":
            # Pearson correlation
            for param, values in param_data.items():
                if len(set(values)) > 1:  # Skip constant parameters
                    correlation = np.corrcoef(values, scores)[0, 1]
                    importance_scores[param] = abs(correlation)
        
        elif method == "variance":
            # Variance-based importance
            for param, values in param_data.items():
                if len(set(values)) > 1:
                    # Group by parameter value and calculate score variance
                    unique_values = list(set(values))
                    group_variances = []
                    
                    for val in unique_values:
                        group_scores = [scores[i] for i, v in enumerate(values) if v == val]
                        if len(group_scores) > 1:
                            group_variances.append(np.var(group_scores))
                    
                    if group_variances:
                        importance_scores[param] = np.mean(group_variances)
        
        elif method == "mutual_information" and SCIPY_AVAILABLE:
            # Mutual information
            from sklearn.feature_selection import mutual_info_regression
            
            for param, values in param_data.items():
                if len(set(values)) > 1:
                    mi = mutual_info_regression(
                        np.array(values).reshape(-1, 1), 
                        scores, 
                        random_state=42
                    )[0]
                    importance_scores[param] = mi
        
        else:
            # Default to correlation
            for param, values in param_data.items():
                if len(set(values)) > 1:
                    correlation = np.corrcoef(values, scores)[0, 1]
                    importance_scores[param] = abs(correlation)
        
        # Sort by importance
        sorted_params = sorted(importance_scores.items(), 
                             key=lambda x: x[1], reverse=True)[:top_k]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot of importance
        params = [p[0] for p in sorted_params]
        importances = [p[1] for p in sorted_params]
        
        bars = ax1.barh(params, importances)
        
        # Color bars by parameter type
        colors = []
        for param in params:
            if 'lora' in param:
                colors.append('orange')
            elif param in ['guidance', 'steps']:
                colors.append('blue')
            elif param in ['scheduler', 'sampler']:
                colors.append('green')
            else:
                colors.append('gray')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_xlabel(f'Importance ({method})')
        ax1.set_title('Parameter Importance Analysis')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Scatter plot matrix for top 3 parameters
        if len(sorted_params) >= 3:
            top_3_params = [p[0] for p in sorted_params[:3]]
            
            # Create scatter matrix
            for i, param1 in enumerate(top_3_params):
                for j, param2 in enumerate(top_3_params):
                    if i < j:
                        ax2.scatter(param_data[param1], param_data[param2], 
                                  c=scores, cmap='viridis', alpha=0.6)
            
            ax2.set_title('Top Parameter Interactions')
            ax2.set_xlabel(top_3_params[0])
            ax2.set_ylabel(top_3_params[1] if len(top_3_params) > 1 else '')
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close()
        
        # Create summary
        summary = f"Parameter Importance Analysis ({method})\n"
        summary += "=" * 40 + "\n"
        for param, importance in sorted_params:
            summary += f"{param:<20} {importance:.4f}\n"
        
        summary += f"\nMost important: {sorted_params[0][0]}"
        summary += f"\nLeast important: {sorted_params[-1][0]}"
        
        return (torch.from_numpy(img_array).float() / 255.0, summary)

class OptimizationReport:
    """Generates a comprehensive optimization report"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
                "export_format": (["html", "pdf", "markdown", "json"],),
                "report_name": ("STRING", {"default": "optimization_report"}),
            },
            "optional": {
                "include_plots": ("BOOLEAN", {"default": True}),
                "include_history": ("BOOLEAN", {"default": True}),
                "company_name": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report_path", "report_summary")
    FUNCTION = "generate_report"
    CATEGORY = "Bayesian Optimization/Export"
    
    def generate_report(self, config, export_format, report_name, 
                       include_plots=True, include_history=True, 
                       company_name=""):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"{report_name}_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report content
        if export_format == "html":
            report_path = self._generate_html_report(
                config, report_dir, include_plots, include_history, company_name
            )
        elif export_format == "markdown":
            report_path = self._generate_markdown_report(
                config, report_dir, include_plots, include_history, company_name
            )
        elif export_format == "json":
            report_path = self._generate_json_report(config, report_dir)
        else:
            # Default to markdown
            report_path = self._generate_markdown_report(
                config, report_dir, include_plots, include_history, company_name
            )
        
        # Create summary
        summary = f"Optimization Report Generated\n"
        summary += f"Format: {export_format}\n"
        summary += f"Location: {report_path}\n"
        summary += f"Total Iterations: {config['iteration']}\n"
        summary += f"Best Score: {config['best_score']:.4f}\n"
        
        if config["best_params"]:
            summary += "\nBest Parameters:\n"
            for key, value in config["best_params"].items():
                if not key.startswith('_'):
                    summary += f"  {key}: {value}\n"
        
        return (report_path, summary)
    
    def _generate_html_report(self, config, report_dir, include_plots, 
                             include_history, company_name):
        """Generate HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bayesian Optimization Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .metric {{
            background-color: #e8f4f8;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .parameter {{
            background-color: #f0f0f0;
            padding: 5px 10px;
            margin: 5px 0;
            border-radius: 3px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Bayesian Optimization Report</h1>
    {'<p>Generated for: ' + company_name + '</p>' if company_name else ''}
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Summary</h2>
    <div class="metric">
        <strong>Total Iterations:</strong> {config['iteration']}<br>
        <strong>Best Score:</strong> {config['best_score']:.4f}<br>
        <strong>Optimization Method:</strong> {config['similarity_metric']}<br>
        <strong>Initial Points:</strong> {config['n_initial_points']}
    </div>
    
    <h2>Best Parameters</h2>
"""
        
        if config["best_params"]:
            for key, value in config["best_params"].items():
                if not key.startswith('_'):
                    html_content += f'<div class="parameter"><strong>{key}:</strong> {value}</div>\n'
        
        html_content += """
    <h2>Configuration</h2>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Min</th>
            <th>Max</th>
        </tr>
"""
        
        # Add parameter ranges
        if hasattr(config["space"], '__iter__'):
            for item in config["space"]:
                if hasattr(item, 'name') and hasattr(item, 'low') and hasattr(item, 'high'):
                    html_content += f"""
        <tr>
            <td>{item.name}</td>
            <td>{item.low}</td>
            <td>{item.high}</td>
        </tr>
"""
        
        html_content += "</table>\n"
        
        if include_history and config["history"]:
            html_content += """
    <h2>Optimization History</h2>
    <table>
        <tr>
            <th>Iteration</th>
            <th>Score</th>
            <th>Guidance</th>
            <th>Steps</th>
            <th>Scheduler</th>
            <th>Sampler</th>
        </tr>
"""
            for entry in config["history"][-20:]:  # Last 20 entries
                params = entry["params"]
                html_content += f"""
        <tr>
            <td>{entry['iteration']}</td>
            <td>{entry['score']:.4f}</td>
            <td>{params.get('guidance', 'N/A')}</td>
            <td>{params.get('steps', 'N/A')}</td>
            <td>{params.get('scheduler', 'N/A')}</td>
            <td>{params.get('sampler', 'N/A')}</td>
        </tr>
"""
            html_content += "</table>\n"
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML
        html_path = os.path.join(report_dir, "report.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_markdown_report(self, config, report_dir, include_plots, 
                                 include_history, company_name):
        """Generate Markdown report"""
        md_content = f"""# Bayesian Optimization Report

{'**Generated for:** ' + company_name + '\n' if company_name else ''}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

- **Total Iterations:** {config['iteration']}
- **Best Score:** {config['best_score']:.4f}
- **Optimization Method:** {config['similarity_metric']}
- **Initial Points:** {config['n_initial_points']}

## Best Parameters

"""
        
        if config["best_params"]:
            for key, value in config["best_params"].items():
                if not key.startswith('_'):
                    md_content += f"- **{key}:** {value}\n"
        
        md_content += "\n## Fixed Configuration\n\n"
        md_content += f"**Prompt:** {config['fixed_prompt'][:200]}...\n\n"
        
        if include_history and config["history"]:
            md_content += "## Top 10 Results\n\n"
            md_content += "| Iteration | Score | Guidance | Steps | Scheduler | Sampler |\n"
            md_content += "|-----------|-------|----------|-------|-----------|----------|\n"
            
            # Sort by score and show top 10
            sorted_history = sorted(config["history"], 
                                  key=lambda x: x["score"], reverse=True)[:10]
            
            for entry in sorted_history:
                params = entry["params"]
                md_content += f"| {entry['iteration']} "
                md_content += f"| {entry['score']:.4f} "
                md_content += f"| {params.get('guidance', 'N/A')} "
                md_content += f"| {params.get('steps', 'N/A')} "
                md_content += f"| {params.get('scheduler', 'N/A')} "
                md_content += f"| {params.get('sampler', 'N/A')} |\n"
        
        # Save markdown
        md_path = os.path.join(report_dir, "report.md")
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        return md_path
    
    def _generate_json_report(self, config, report_dir):
        """Generate JSON report"""
        report_data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "total_iterations": config["iteration"],
                "best_score": float(config["best_score"]),
                "optimization_method": config["similarity_metric"],
            },
            "best_parameters": config["best_params"],
            "configuration": {
                "n_iterations": config["n_iterations"],
                "n_initial_points": config["n_initial_points"],
                "fixed_prompt": config["fixed_prompt"],
            },
            "history": config["history"]
        }
        
        # Save JSON
        json_path = os.path.join(report_dir, "report.json")
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return json_path

class ParameterRecommendation:
    """Provides parameter recommendations based on optimization results"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ENHANCED_BAYESIAN_CONFIG",),
                "recommendation_type": (["conservative", "balanced", "aggressive", "top_k"],),
            },
            "optional": {
                "k": ("INT", {"default": 3, "min": 1, "max": 10}),
                "confidence_level": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 0.99}),
            }
        }
    
    RETURN_TYPES = ("STRING", "PARAMETER_SET")
    RETURN_NAMES = ("recommendations", "parameter_set")
    FUNCTION = "generate_recommendations"
    CATEGORY = "Bayesian Optimization/Export"
    
    def generate_recommendations(self, config, recommendation_type, 
                               k=3, confidence_level=0.95):
        
        if not config["history"]:
            return ("No optimization history available", {})
        
        recommendations = f"Parameter Recommendations ({recommendation_type})\n"
        recommendations += "=" * 50 + "\n\n"
        
        if recommendation_type == "conservative":
            # Recommend parameters close to best with small variance
            if config["best_params"]:
                param_set = config["best_params"].copy()
                
                recommendations += "Conservative recommendation (minimal risk):\n"
                recommendations += "Based on best found parameters with safety margins\n\n"
                
                for key, value in param_set.items():
                    if not key.startswith('_'):
                        recommendations += f"{key}: {value}\n"
                
                recommendations += "\nNote: These parameters have shown consistent good performance"
                
        elif recommendation_type == "balanced":
            # Recommend weighted average of top performers
            sorted_history = sorted(config["history"], 
                                  key=lambda x: x["score"], reverse=True)
            top_entries = sorted_history[:min(5, len(sorted_history))]
            
            param_set = {}
            param_weights = {}
            total_weight = 0
            
            for i, entry in enumerate(top_entries):
                weight = (len(top_entries) - i) / len(top_entries)
                total_weight += weight
                
                for key, value in entry["params"].items():
                    if isinstance(value, (int, float)):
                        if key not in param_set:
                            param_set[key] = 0
                            param_weights[key] = 0
                        param_set[key] += value * weight
                        param_weights[key] += weight
                    elif key not in param_set:
                        # For categorical, use the best performer's value
                        param_set[key] = value
            
            # Normalize continuous parameters
            for key in param_weights:
                if param_weights[key] > 0:
                    param_set[key] = param_set[key] / param_weights[key]
                    if key == "steps":
                        param_set[key] = int(round(param_set[key]))
            
            recommendations += "Balanced recommendation (weighted average of top 5):\n\n"
            for key, value in param_set.items():
                if not key.startswith('_'):
                    if isinstance(value, float):
                        recommendations += f"{key}: {value:.3f}\n"
                    else:
                        recommendations += f"{key}: {value}\n"
            
        elif recommendation_type == "aggressive":
            # Recommend exploration beyond best found
            if config["best_params"] and SCIPY_AVAILABLE:
                param_set = config["best_params"].copy()
                
                # Calculate parameter gradients near best point
                nearby_entries = []
                for entry in config["history"]:
                    # Check if parameters are close to best
                    distance = 0
                    for key in ['guidance', 'steps']:
                        if key in entry["params"] and key in config["best_params"]:
                            distance += abs(entry["params"][key] - 
                                         config["best_params"][key])
                    
                    if distance < 5:  # Arbitrary threshold
                        nearby_entries.append(entry)
                
                if len(nearby_entries) > 3:
                    # Extrapolate in direction of improvement
                    recommendations += "Aggressive recommendation (extrapolated):\n"
                    recommendations += "Pushing parameters in direction of improvement\n\n"
                    
                    # Simple gradient estimation
                    for key in ['guidance', 'steps']:
                        if key in param_set:
                            values = [e["params"].get(key, param_set[key]) 
                                    for e in nearby_entries]
                            scores = [e["score"] for e in nearby_entries]
                            
                            if len(set(values)) > 1:
                                # Calculate correlation
                                corr = np.corrcoef(values, scores)[0, 1]
                                
                                # Adjust parameter in direction of positive correlation
                                adjustment = 0.1 * np.std(values) * np.sign(corr)
                                param_set[key] = param_set[key] + adjustment
                                
                                if key == "steps":
                                    param_set[key] = int(round(param_set[key]))
                else:
                    recommendations += "Using best parameters (insufficient data for extrapolation)\n\n"
                
                for key, value in param_set.items():
                    if not key.startswith('_'):
                        if isinstance(value, float):
                            recommendations += f"{key}: {value:.3f}\n"
                        else:
                            recommendations += f"{key}: {value}\n"
            else:
                param_set = config["best_params"] or {}
                recommendations += "Using best found parameters\n"
                
        else:  # top_k
            # Recommend top k parameter sets
            sorted_history = sorted(config["history"], 
                                  key=lambda x: x["score"], reverse=True)
            top_entries = sorted_history[:min(k, len(sorted_history))]
            
            recommendations += f"Top {k} parameter sets:\n\n"
            
            param_sets = []
            for i, entry in enumerate(top_entries):
                recommendations += f"Rank {i+1} (Score: {entry['score']:.4f}):\n"
                param_set = entry["params"].copy()
                param_sets.append(param_set)
                
                for key, value in param_set.items():
                    if not key.startswith('_'):
                        recommendations += f"  {key}: {value}\n"
                recommendations += "\n"
            
            # Return the best one as the main parameter set
            param_set = param_sets[0] if param_sets else {}
        
        # Add confidence intervals if possible
        if config["history"] and len(config["history"]) > 10:
            recommendations += "\n\nParameter Ranges (based on top 20%):\n"
            
            top_20_percent = sorted(config["history"], 
                                  key=lambda x: x["score"], 
                                  reverse=True)[:max(1, len(config["history"]) // 5)]
            
            for param in ['guidance', 'steps']:
                values = [e["params"].get(param) for e in top_20_percent 
                         if param in e["params"]]
                if values:
                    recommendations += f"{param}: {min(values):.2f} - {max(values):.2f}\n"
        
        return (recommendations, param_set)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ParameterHeatmap": ParameterHeatmap,
    "ConvergencePlot": ConvergencePlot,
    "ParameterImportanceAnalysis": ParameterImportanceAnalysis,
    "OptimizationReport": OptimizationReport,
    "ParameterRecommendation": ParameterRecommendation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParameterHeatmap": "Parameter Heatmap",
    "ConvergencePlot": "Convergence Plot",
    "ParameterImportanceAnalysis": "Parameter Importance Analysis",
    "OptimizationReport": "Optimization Report Generator",
    "ParameterRecommendation": "Parameter Recommendation",
}