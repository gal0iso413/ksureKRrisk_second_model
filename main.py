"""
Main execution script for the second model.

This script demonstrates how to use the two-stage model for risk prediction
and duration estimation.
"""

import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocessing import preprocess_pipeline
from models.two_stage_model import TwoStageModel
from src.utils.logging_config import setup_logger
from src.utils.common import validate_dataframe

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data/raw",
        "data/processed",
        "logs",
        "models/saved"
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
        'days_with_alarm',  # Target variable
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
        
        # Load and preprocess data
        input_data = load_data("data/raw/input_data.csv")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        processed_data = preprocess_pipeline(
            input_path=input_data,
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
            target_column='days_with_alarm',
            date_column='date',
            identifier_columns=['company_id'],
            group_by='company_id'
        )
        
        # Prepare features and targets
        X = processed_data.drop(['days_with_alarm', 'date', 'company_id'], axis=1)
        y1 = (processed_data['days_with_alarm'] > 0).astype(int)  # Binary classification target
        y2 = processed_data['days_with_alarm']  # Regression target
        
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
        
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 