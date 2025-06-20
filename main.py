"""
Main execution script for the second model.

This script demonstrates how to use the two-stage model for risk prediction
and duration estimation with comprehensive preprocessing and visualization.
"""

# Set matplotlib backend to avoid Qt issues
import matplotlib
matplotlib.use('agg')  # Use non-interactive backend

import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocessing import preprocess_pipeline
from models.two_stage_model import TwoStageModel
from src.utils.logging_config import setup_logger
from src.utils.common import validate_dataframe
from src.visualization import create_visualization_pipeline

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data/raw",
        "data/processed",
        "logs",
        "models/saved",
        "results"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Set up logging
logger = setup_logger(
    name=__name__,
    log_file=Path("logs/main.log")
)

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load and validate input data.
    
    Args:
        data_path: Path to input data file
        
    Returns:
        Loaded and validated DataFrame
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_columns = [
        'date',
        'company_id',
        'alarm_type',   # Stage 1 target: classification
        'alarm_day',    # Stage 2 target: regression
        # Add other required columns here
    ]
    
    validate_dataframe(
        df,
        required_columns=required_columns,
        date_columns=['date']
    )
    
    return df

def main():
    """Main execution function."""
    try:
        # Create necessary directories
        setup_directories()
        
        # Preprocess data with outlier handling
        logger.info("Preprocessing data with outlier handling...")
        processed_data = preprocess_pipeline(
            input_path="data/raw/input_data.csv",
            output_path="data/processed/processed_data.csv",
            numeric_columns=[
                # Add your numeric columns here
                'feature1',
                'feature2'
            ],
            categorical_columns=[
                # Add your categorical columns here
                'category1',
                'category2'
            ],
            target_column='alarm_day',  # Stage 2 regression target
            date_column='date',
            identifier_columns=['company_id'],
            group_by='company_id',
            handle_outliers=True,  # Enable outlier handling
            outlier_columns=None   # Use all numeric columns for outlier handling
        )
        
        # Prepare features and targets
        # Note: After preprocessing, categorical columns will be one-hot encoded
        # Only drop columns that are guaranteed to exist
        columns_to_drop = ['alarm_type', 'alarm_day', 'date', 'company_id']
        available_columns = [col for col in columns_to_drop if col in processed_data.columns]
        
        X = processed_data.drop(available_columns, axis=1)
        y1 = processed_data['alarm_type']  # Stage 1: Classification target
        y2 = processed_data['alarm_day']   # Stage 2: Regression target
        
        # Split data
        X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
            X, y1, y2,
            test_size=0.2,
            random_state=42
        )
        
        # Initialize and train model
        logger.info("Training two-stage model...")
        model = TwoStageModel(
            stage1_model_type='xgboost',
            stage2_model_type='xgboost'
        )
        
        model.fit(
            X=X_train,
            y1=y1_train,
            y2=y2_train,
            feature_names=X.columns.tolist()
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = model.evaluate(X_test, y1_test, y2_test)
        
        # Log metrics
        logger.info("Model evaluation metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Get feature importance
        importance = model.get_feature_importance()
        logger.info("\nFeature importance:")
        for stage, features in importance.items():
            logger.info(f"\n{stage.upper()} stage:")
            for feature, score in sorted(features.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"{feature}: {score:.4f}")
        
        # Create comprehensive visualizations
        logger.info("Creating visualizations...")
        create_visualization_pipeline(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y1_train=y1_train,
            y1_test=y1_test,
            y2_train=y2_train,
            y2_test=y2_test,
            metrics=metrics,
            importance_dict=importance,
            output_dir="results",
            model_name="two_stage_model"
        )
        
        logger.info("Model training, evaluation, and visualization completed successfully")
        logger.info(f"Results saved to: {Path('results').absolute()}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 