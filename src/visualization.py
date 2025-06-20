"""
Visualization module for the second model.

This module provides plotting functionality for:
- Model evaluation results
- Feature importance plots
- SHAP analysis plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

# Import SHAP for model interpretability
import shap

from utils.common import create_directory_if_not_exists
from constants import DEFAULT_RANDOM_STATE, KOREAN_FONT_CONFIG, RISK_TYPES

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set up Korean font
plt.rcParams['font.family'] = KOREAN_FONT_CONFIG['font_family']
plt.rcParams['font.size'] = KOREAN_FONT_CONFIG['font_size']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Class for creating model visualization plots."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        create_directory_if_not_exists(self.output_dir, "Results directory")
        
        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "shap").mkdir(exist_ok=True)
        (self.output_dir / "feature_importance").mkdir(exist_ok=True)
    
    def plot_model_metrics(
        self,
        metrics: Dict[str, float],
        model_name: str = "two_stage_model",
        save_plot: bool = True
    ) -> None:
        """
        Plot model evaluation metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            model_name: Name of the model for plot title
            save_plot: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stage 1 metrics (classification)
        stage1_metrics = {k: v for k, v in metrics.items() if k.startswith('stage1_')}
        if stage1_metrics:
            metric_names = [k.replace('stage1_', '').replace('_', ' ').title() for k in stage1_metrics.keys()]
            metric_values = list(stage1_metrics.values())
            
            axes[0].bar(metric_names, metric_values, color='skyblue', alpha=0.7)
            axes[0].set_title(f'{model_name} - Stage 1 (Classification) Metrics', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Score')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(metric_values):
                axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Stage 2 metrics (regression)
        stage2_metrics = {k: v for k, v in metrics.items() if k.startswith('stage2_')}
        if stage2_metrics:
            metric_names = [k.replace('stage2_', '').replace('_', ' ').title() for k in stage2_metrics.keys()]
            metric_values = list(stage2_metrics.values())
            
            axes[1].bar(metric_names, metric_values, color='lightcoral', alpha=0.7)
            axes[1].set_title(f'{model_name} - Stage 2 (Regression) Metrics', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Score')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(metric_values):
                axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / "plots" / f"{model_name}_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model metrics plot to {plot_path}")
        
        plt.close()  # Close the figure instead of showing
    
    def plot_feature_importance(
        self,
        importance_dict: Dict[str, Dict[str, float]],
        top_n: int = 20,
        model_name: str = "two_stage_model",
        save_plot: bool = True
    ) -> None:
        """
        Plot feature importance for both stages.
        
        Args:
            importance_dict: Dictionary with stage1 and stage2 feature importance
            top_n: Number of top features to show
            model_name: Name of the model for plot title
            save_plot: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for i, (stage, importance) in enumerate(importance_dict.items()):
            if not importance:
                continue
                
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, scores = zip(*sorted_features)
            
            # Create horizontal bar plot
            axes[i].barh(range(len(features)), scores, color='lightblue', alpha=0.7)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels(features)
            axes[i].set_xlabel('Importance Score')
            axes[i].set_title(f'{model_name} - {stage.title()} Feature Importance', 
                            fontsize=14, fontweight='bold')
            axes[i].invert_yaxis()
            
            # Add value labels
            for j, score in enumerate(scores):
                axes[i].text(score + 0.001, j, f'{score:.3f}', 
                           ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / "feature_importance" / f"{model_name}_feature_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {plot_path}")
        
        plt.close()  # Close the figure instead of showing
    
    def plot_shap_analysis(
        self,
        model,
        X: Union[pd.DataFrame, np.ndarray],
        stage: str = "stage1",
        model_name: str = "two_stage_model",
        sample_size: int = 100,
        save_plot: bool = True
    ) -> None:
        """
        Create SHAP analysis plots.
        
        Args:
            model: Trained model (XGBoost or LightGBM)
            X: Feature matrix
            stage: Which stage to analyze ('stage1' or 'stage2')
            model_name: Name of the model for plot title
            sample_size: Number of samples to use for SHAP analysis
            save_plot: Whether to save the plot
        """
        try:
            # Sample data if too large
            if len(X) > sample_size:
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]
            else:
                X_sample = X
            
            # Create SHAP explainer
            if hasattr(model, 'booster'):  # XGBoost
                explainer = shap.TreeExplainer(model.booster)
            else:  # LightGBM
                explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, 
                X_sample,
                plot_type="bar",
                show=False
            )
            plt.title(f'{model_name} - {stage.title()} SHAP Summary', 
                     fontsize=16, fontweight='bold', pad=20)
            
            if save_plot:
                plot_path = self.output_dir / "shap" / f"{model_name}_{stage}_shap_summary.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP summary plot to {plot_path}")
            
            plt.close()  # Close the figure instead of showing
            
        except Exception as e:
            logger.error(f"Error creating SHAP plot for {stage}: {str(e)}")
    
    def plot_prediction_vs_actual(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        stage: str = "stage2",
        model_name: str = "two_stage_model",
        save_plot: bool = True
    ) -> None:
        """
        Plot predicted vs actual values for regression stage.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            stage: Which stage ('stage1' or 'stage2')
            model_name: Name of the model for plot title
            save_plot: Whether to save the plot
        """
        if stage == "stage2":  # Regression
            plt.figure(figsize=(10, 8))
            
            # Scatter plot
            plt.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{model_name} - {stage.title()} Prediction vs Actual', 
                     fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            if save_plot:
                plot_path = self.output_dir / "plots" / f"{model_name}_{stage}_pred_vs_actual.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved prediction vs actual plot to {plot_path}")
            
            plt.close()  # Close the figure instead of showing
        
        elif stage == "stage1":  # Classification
            from sklearn.metrics import confusion_matrix
            
            plt.figure(figsize=(8, 6))
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Alarm'],
                       yticklabels=['Normal', 'Alarm'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'{model_name} - {stage.title()} Confusion Matrix', 
                     fontsize=14, fontweight='bold')
            
            if save_plot:
                plot_path = self.output_dir / "plots" / f"{model_name}_{stage}_confusion_matrix.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved confusion matrix plot to {plot_path}")
            
            plt.close()  # Close the figure instead of showing


def create_visualization_pipeline(
    model,
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y1_train: Union[pd.Series, np.ndarray],
    y1_test: Union[pd.Series, np.ndarray],
    y2_train: Union[pd.Series, np.ndarray],
    y2_test: Union[pd.Series, np.ndarray],
    metrics: Dict[str, float],
    importance_dict: Dict[str, Dict[str, float]],
    output_dir: str = "results",
    model_name: str = "two_stage_model"
) -> None:
    """
    Create comprehensive visualization pipeline.
    
    Args:
        model: Trained two-stage model
        X_train, X_test: Training and test feature matrices
        y1_train, y1_test: Training and test classification targets
        y2_train, y2_test: Training and test regression targets
        metrics: Model evaluation metrics
        importance_dict: Feature importance dictionary
        output_dir: Directory to save plots
        model_name: Name of the model
    """
    visualizer = ModelVisualizer(output_dir)
    
    # Plot model metrics
    visualizer.plot_model_metrics(metrics, model_name)
    
    # Plot feature importance
    visualizer.plot_feature_importance(importance_dict, model_name=model_name)
    
    # Plot SHAP analysis for both stages
    if hasattr(model, 'stage1_model') and model.stage1_model is not None:
        visualizer.plot_shap_analysis(
            model.stage1_model, X_test, "stage1", model_name
        )
    
    if hasattr(model, 'stage2_model') and model.stage2_model is not None:
        visualizer.plot_shap_analysis(
            model.stage2_model, X_test, "stage2", model_name
        )
    
    # Plot prediction vs actual
    y1_pred, y2_pred = model.predict(X_test)
    
    # Only plot for non-normal cases in stage 2
    non_normal_mask = y1_test != 'normal'
    if non_normal_mask.any():
        visualizer.plot_prediction_vs_actual(
            y2_test[non_normal_mask], y2_pred[non_normal_mask], 
            "stage2", model_name
        )
    
    # Plot confusion matrix for stage 1
    visualizer.plot_prediction_vs_actual(
        y1_test, y1_pred, "stage1", model_name
    )
    
    logger.info(f"Visualization pipeline completed. Plots saved to {output_dir}") 