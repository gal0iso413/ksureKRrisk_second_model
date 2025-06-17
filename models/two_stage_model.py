"""
Two-stage model for risk prediction and duration estimation.

This module implements a two-stage modeling approach:
1. First stage: Classification model to predict risk type
2. Second stage: Regression model to predict duration for non-normal cases
"""

import logging
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from ..src.constants import (
    MODEL_TYPES,
    XGBOOST_CLASSIFIER_PARAMS,
    XGBOOST_REGRESSOR_PARAMS,
    LIGHTGBM_CLASSIFIER_PARAMS,
    LIGHTGBM_REGRESSOR_PARAMS
)

logger = logging.getLogger(__name__)

class TwoStageModel:
    """Two-stage model for risk prediction and duration estimation."""
    
    def __init__(
        self,
        stage1_model_type: str = 'xgboost',
        stage2_model_type: str = 'xgboost',
        stage1_params: Optional[Dict] = None,
        stage2_params: Optional[Dict] = None
    ):
        """
        Initialize the two-stage model.
        
        Args:
            stage1_model_type: Type of model for stage 1 (classification)
            stage2_model_type: Type of model for stage 2 (regression)
            stage1_params: Optional parameters for stage 1 model
            stage2_params: Optional parameters for stage 2 model
        """
        if stage1_model_type not in MODEL_TYPES:
            raise ValueError(f"Invalid stage1_model_type. Must be one of {MODEL_TYPES}")
        if stage2_model_type not in MODEL_TYPES:
            raise ValueError(f"Invalid stage2_model_type. Must be one of {MODEL_TYPES}")
            
        self.stage1_model_type = stage1_model_type
        self.stage2_model_type = stage2_model_type
        self.stage1_params = stage1_params or self._get_default_params(stage1_model_type, is_classifier=True)
        self.stage2_params = stage2_params or self._get_default_params(stage2_model_type, is_classifier=False)
        
        self.stage1_model = None
        self.stage2_model = None
        self.feature_names = None
        
    def _get_default_params(self, model_type: str, is_classifier: bool) -> Dict:
        """Get default parameters for the specified model type."""
        if model_type == 'xgboost':
            return XGBOOST_CLASSIFIER_PARAMS if is_classifier else XGBOOST_REGRESSOR_PARAMS
        elif model_type == 'lightgbm':
            return LIGHTGBM_CLASSIFIER_PARAMS if is_classifier else LIGHTGBM_REGRESSOR_PARAMS
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def _create_model(self, model_type: str, is_classifier: bool) -> BaseEstimator:
        """Create a model instance based on the specified type."""
        if model_type == 'xgboost':
            if is_classifier:
                return xgb.XGBClassifier(**self.stage1_params)
            else:
                return xgb.XGBRegressor(**self.stage2_params)
        elif model_type == 'lightgbm':
            if is_classifier:
                return lgb.LGBMClassifier(**self.stage1_params)
            else:
                return lgb.LGBMRegressor(**self.stage2_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y1: Union[pd.Series, np.ndarray],
        y2: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> 'TwoStageModel':
        """
        Fit both stages of the model.
        
        Args:
            X: Feature matrix
            y1: Risk type labels (classification target)
            y2: Duration values (regression target)
            feature_names: Optional list of feature names
            
        Returns:
            self: The fitted model instance
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
        # Stage 1: Train classification model
        logger.info("Training stage 1 (classification) model...")
        self.stage1_model = self._create_model(self.stage1_model_type, is_classifier=True)
        self.stage1_model.fit(X, y1)
        
        # Stage 2: Train regression model only on non-normal cases
        logger.info("Training stage 2 (regression) model...")
        non_normal_mask = y1 != 'normal'
        if non_normal_mask.any():
            self.stage2_model = self._create_model(self.stage2_model_type, is_classifier=False)
            self.stage2_model.fit(X[non_normal_mask], y2[non_normal_mask])
        else:
            logger.warning("No non-normal cases found for stage 2 training")
            self.stage2_model = None
            
        return self
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using both stages.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (risk_types, durations)
        """
        # Stage 1: Predict risk types
        risk_types = self.stage1_model.predict(X)
        
        # Stage 2: Predict durations only for non-normal cases
        durations = np.zeros(len(X))
        non_normal_mask = risk_types != 'normal'
        
        if non_normal_mask.any() and self.stage2_model is not None:
            durations[non_normal_mask] = self.stage2_model.predict(X[non_normal_mask])
            
        return risk_types, durations
        
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y1_true: Union[pd.Series, np.ndarray],
        y2_true: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate both stages of the model.
        
        Args:
            X: Feature matrix
            y1_true: True risk type labels
            y2_true: True duration values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y1_pred, y2_pred = self.predict(X)
        
        # Stage 1 metrics
        metrics = {
            'stage1_f1': f1_score(y1_true, y1_pred, average='weighted'),
            'stage1_precision': precision_score(y1_true, y1_pred, average='weighted'),
            'stage1_recall': recall_score(y1_true, y1_pred, average='weighted')
        }
        
        # Stage 2 metrics (only on non-normal cases)
        non_normal_mask = y1_true != 'normal'
        if non_normal_mask.any():
            metrics.update({
                'stage2_rmse': np.sqrt(mean_squared_error(y2_true[non_normal_mask], y2_pred[non_normal_mask])),
                'stage2_mae': mean_absolute_error(y2_true[non_normal_mask], y2_pred[non_normal_mask])
            })
            
        return metrics
        
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from both stages.
        
        Returns:
            Dictionary containing feature importance for both stages
        """
        importance = {}
        
        # Stage 1 feature importance
        if hasattr(self.stage1_model, 'feature_importances_'):
            importance['stage1'] = dict(zip(self.feature_names, self.stage1_model.feature_importances_))
            
        # Stage 2 feature importance
        if self.stage2_model is not None and hasattr(self.stage2_model, 'feature_importances_'):
            importance['stage2'] = dict(zip(self.feature_names, self.stage2_model.feature_importances_))
            
        return importance 