"""
Data preprocessing module for the two-stage risk prediction model.

This module provides comprehensive data preprocessing functionality including:
- Regression-specific preprocessing
- Zero-inflated data handling
- Feature engineering for time series data
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Import project utilities
from src.utils.logging_config import get_logger
from src.utils.common import (
    safe_load_csv, safe_save_csv, clean_column_names,
    standardize_error_handling, create_directory_if_not_exists
)
from src.constants import (
    DEFAULT_RANDOM_STATE, DEFAULT_SCALER_TYPE,
    DEFAULT_ENCODING_METHOD, DEFAULT_ENCODING
)

# Initialize logger
logger = get_logger(__name__)

class RegressionDataProcessor:
    """Class for regression-specific data preprocessing."""
    
    def __init__(
        self,
        scaler_type: str = DEFAULT_SCALER_TYPE,
        random_state: int = DEFAULT_RANDOM_STATE
    ):
        """
        Initialize data processor.
        
        Args:
            scaler_type: Type of scaler to use ('standard' or 'minmax')
            random_state: Random seed for reproducibility
        """
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
    
    @standardize_error_handling
    def preprocess_features(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        target_column: str,
        identifier_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Preprocess features for regression model.
        
        Args:
            df: Input DataFrame
            numeric_columns: List of numeric columns
            categorical_columns: List of categorical columns
            target_column: Name of target column
            identifier_columns: List of identifier columns to preserve
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Clean column names
        df = clean_column_names(df)
        
        # Handle identifier columns
        if identifier_columns:
            identifier_data = df[identifier_columns].copy()
        
        # Process numeric features
        if numeric_columns:
            df = self._process_numeric_features(df, numeric_columns)
        
        # Process categorical features
        if categorical_columns:
            df = self._process_categorical_features(df, categorical_columns)
        
        # Process target variable
        if target_column in df.columns:
            df = self._process_target_variable(df, target_column)
        
        # Restore identifier columns
        if identifier_columns:
            for col in identifier_columns:
                if col in identifier_data.columns:
                    df[col] = identifier_data[col]
        
        return df
    
    def _process_numeric_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Process numeric features."""
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        df[columns] = imputer.fit_transform(df[columns])
        self.imputers['numeric'] = imputer
        
        # Scale features
        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        df[columns] = scaler.fit_transform(df[columns])
        self.scalers['numeric'] = scaler
        
        return df
    
    def _process_categorical_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Process categorical features."""
        # Handle missing values
        imputer = SimpleImputer(strategy='most_frequent')
        df[columns] = imputer.fit_transform(df[columns])
        self.imputers['categorical'] = imputer
        
        # One-hot encoding
        df = pd.get_dummies(df, columns=columns, prefix=columns)
        
        return df
    
    def _process_target_variable(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> pd.DataFrame:
        """Process target variable for zero-inflated data."""
        # Log target distribution
        zero_count = (df[target_column] == 0).sum()
        total_count = len(df)
        zero_ratio = zero_count / total_count
        
        logger.info(f"Target variable distribution:")
        logger.info(f"- Total samples: {total_count}")
        logger.info(f"- Zero values: {zero_count} ({zero_ratio:.2%})")
        logger.info(f"- Non-zero values: {total_count - zero_count} ({1 - zero_ratio:.2%})")
        
        # Log non-zero value statistics
        non_zero_values = df[df[target_column] > 0][target_column]
        if len(non_zero_values) > 0:
            logger.info(f"Non-zero value statistics:")
            logger.info(f"- Mean: {non_zero_values.mean():.2f}")
            logger.info(f"- Median: {non_zero_values.median():.2f}")
            logger.info(f"- Std: {non_zero_values.std():.2f}")
            logger.info(f"- Min: {non_zero_values.min():.2f}")
            logger.info(f"- Max: {non_zero_values.max():.2f}")
        
        return df
    
    def create_time_features(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        
        # Convert to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Extract time features
        df['year'] = df[date_column].dt.year
        df['quarter'] = df[date_column].dt.quarter
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        
        # Create year-over-year features
        if 'year' in df.columns:
            df['year_since_start'] = df['year'] - df['year'].min()
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        group_by: Union[str, List[str]],
        lags: List[int] = [1, 2, 3]
    ) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            group_by: Column or list of columns to group by (e.g., 'company_id' or ['company_id', 'region'])
            lags: List of lag periods
        """
        df = df.copy()
        # Ensure group_by is a list for sorting
        sort_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        df = df.sort_values(sort_cols + ['year'])
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby(group_by)[col].shift(lag)
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        group_by: Union[str, List[str]],
        windows: List[int] = [2, 3]
    ) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            group_by: Column or list of columns to group by
            windows: List of window sizes
        """
        df = df.copy()
        sort_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        df = df.sort_values(sort_cols + ['year'])
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df.groupby(group_by)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{col}_rolling_std_{window}'] = df.groupby(group_by)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
        return df


@standardize_error_handling
def preprocess_pipeline(
    input_path: str,
    output_path: str,
    numeric_columns: List[str],
    categorical_columns: List[str],
    target_column: str,
    date_column: Optional[str] = None,
    identifier_columns: Optional[List[str]] = None,
    group_by: Optional[Union[str, List[str]]] = None,
    create_time_features: bool = True,
    create_lag_features: bool = True,
    create_rolling_features: bool = True
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed CSV file
        numeric_columns: List of numeric columns
        categorical_columns: List of categorical columns
        target_column: Name of target column
        date_column: Name of date column
        identifier_columns: List of identifier columns
        group_by: Column or list of columns to group by for time series features
        create_time_features: Whether to create time features
        create_lag_features: Whether to create lag features
        create_rolling_features: Whether to create rolling features
    """
    # Read data
    df = safe_load_csv(input_path, encoding=DEFAULT_ENCODING)
    logger.info(f"Loaded data with shape {df.shape}")
    
    # Initialize processor
    processor = RegressionDataProcessor()
    
    # Basic preprocessing
    df = processor.preprocess_features(
        df=df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        identifier_columns=identifier_columns
    )
    
    # Create time features
    if create_time_features and date_column:
        df = processor.create_time_features(df, date_column)
    
    # Create lag features
    if create_lag_features and group_by:
        df = processor.create_lag_features(
            df=df,
            columns=numeric_columns,
            group_by=group_by
        )
    
    # Create rolling features
    if create_rolling_features and group_by:
        df = processor.create_rolling_features(
            df=df,
            columns=numeric_columns,
            group_by=group_by
        )
    
    # Save processed data
    safe_save_csv(df, output_path, encoding=DEFAULT_ENCODING)
    return df
    logger.info(f"Preprocessing completed. Final shape: {df.shape}") 