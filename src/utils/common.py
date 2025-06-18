"""
Common utilities for the second model.

This module provides common utility functions used across the second model,
including error handling, data validation, and file operations.
"""

import functools
import logging
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union
import numpy as np
import pandas as pd

# Type variable for decorator
T = TypeVar('T')

def standardize_error_handling(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to standardize error handling across functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with standardized error handling
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(func.__module__)
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def create_directory_if_not_exists(
    path: Union[str, Path],
    description: str = "Directory"
) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
        description: Description of directory for logging
        
    Returns:
        Path object for created/existing directory
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.info(f"Created {description}: {path}")
    return path

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    numeric_columns: Optional[list] = None,
    categorical_columns: Optional[list] = None,
    date_columns: Optional[list] = None
) -> None:
    """
    Validate DataFrame structure and data types.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        numeric_columns: List of columns that should be numeric
        categorical_columns: List of columns that should be categorical
        date_columns: List of columns that should be datetime
        
    Raises:
        ValueError: If validation fails
    """
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check numeric columns
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if not np.issubdtype(df[col].dtype, np.number)]
        if non_numeric:
            raise ValueError(f"Non-numeric columns found: {non_numeric}")
    
    # Check categorical columns
    if categorical_columns:
        non_categorical = [col for col in categorical_columns if not pd.api.types.is_categorical_dtype(df[col])]
        if non_categorical:
            raise ValueError(f"Non-categorical columns found: {non_categorical}")
    
    # Check date columns
    if date_columns:
        non_date = [col for col in date_columns if not pd.api.types.is_datetime64_dtype(df[col])]
        if non_date:
            raise ValueError(f"Non-datetime columns found: {non_date}")

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default value if division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Default value to return if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        return a / b
    except ZeroDivisionError:
        return default

def calculate_percentage_change(
    current: float,
    previous: float,
    default: float = 0.0
) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        current: Current value
        previous: Previous value
        default: Default value to return if calculation fails
        
    Returns:
        Percentage change or default value
    """
    if previous == 0:
        return default
    return ((current - previous) / previous) * 100

def format_number(
    number: float,
    decimals: int = 2,
    percentage: bool = False
) -> str:
    """
    Format number with specified decimal places and optional percentage.
    
    Args:
        number: Number to format
        decimals: Number of decimal places
        percentage: Whether to format as percentage
        
    Returns:
        Formatted number string
    """
    if percentage:
        return f"{number:.{decimals}f}%"
    return f"{number:.{decimals}f}"

def safe_load_csv(file_path, encoding="utf-8", **kwargs):
    """Safely load a CSV file and return a DataFrame."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    try:
        return pd.read_csv(file_path, encoding=encoding, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

def safe_save_csv(df, file_path, encoding="utf-8", **kwargs):
    """Safely save a DataFrame to a CSV file."""
    file_path = Path(file_path)
    try:
        df.to_csv(file_path, encoding=encoding, index=False, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to save CSV: {e}")

def clean_column_names(df):
    """Clean DataFrame column names by stripping whitespace and lowering case."""
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
    return df 