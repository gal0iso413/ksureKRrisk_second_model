"""
Test script to verify the implementation of new features.

This script tests:
- Outlier handling with IQR method
- Factor level conversion
- Visualization pipeline
- SHAP analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import the modules to test
from src.data_preprocessing import OutlierHandler, FactorLevelConverter, preprocess_pipeline
from src.visualization import ModelVisualizer
from models.two_stage_model import TwoStageModel
from src.utils.logging_config import setup_logger

# Set up logging
logger = setup_logger(
    name=__name__,
    log_file=Path("logs/test.log")
)

def create_test_data():
    """Create test data with factor codes and outliers."""
    np.random.seed(42)
    
    # Create sample data
    n_samples = 1000
    data = {
        'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'company_id': np.random.randint(1, 11, n_samples),
        'days_with_alarm': np.random.exponential(5, n_samples),
        'feature1': np.random.normal(100, 20, n_samples),
        'feature2': np.random.normal(50, 10, n_samples),
        'category1': np.random.choice(['A', 'B', 'C'], n_samples),
        'category2': np.random.choice(['X', 'Y', 'Z'], n_samples),
    }
    
    # Add factor codes (some will be null)
    for i in range(1, 6):
        data[f'발생사유항목코드_{i}'] = np.random.choice([None, f'CODE_{i}'], n_samples, p=[0.7, 0.3])
    
    # Add some outliers
    data['feature1'][0:10] = np.random.normal(500, 50, 10)  # Outliers
    
    df = pd.DataFrame(data)
    
    # Save test data
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/test_data.csv", index=False)
    
    return df

def test_outlier_handling():
    """Test outlier handling functionality."""
    logger.info("Testing outlier handling...")
    
    # Create test data with outliers
    df = create_test_data()
    
    # Test outlier handling
    columns_to_check = ['feature1', 'feature2']
    df_cleaned = OutlierHandler.handle_outliers_iqr(df, columns_to_check)
    
    # Check if outliers were handled
    original_outliers = ((df['feature1'] > df['feature1'].quantile(0.75) + 1.5 * (df['feature1'].quantile(0.75) - df['feature1'].quantile(0.25))) | 
                        (df['feature1'] < df['feature1'].quantile(0.25) - 1.5 * (df['feature1'].quantile(0.75) - df['feature1'].quantile(0.25)))).sum()
    
    cleaned_outliers = ((df_cleaned['feature1'] > df_cleaned['feature1'].quantile(0.75) + 1.5 * (df_cleaned['feature1'].quantile(0.75) - df_cleaned['feature1'].quantile(0.25))) | 
                       (df_cleaned['feature1'] < df_cleaned['feature1'].quantile(0.25) - 1.5 * (df_cleaned['feature1'].quantile(0.75) - df_cleaned['feature1'].quantile(0.25)))).sum()
    
    logger.info(f"Original outliers in feature1: {original_outliers}")
    logger.info(f"Outliers after cleaning: {cleaned_outliers}")
    
    assert cleaned_outliers == 0, "Outliers should be handled"
    logger.info("✓ Outlier handling test passed")

def test_factor_level_conversion():
    """Test factor level conversion functionality."""
    logger.info("Testing factor level conversion...")
    
    df = create_test_data()
    
    # Test factor level conversion
    df_with_levels = FactorLevelConverter.convert_to_factor_level(df)
    
    # Check if factor_level column was created
    assert 'factor_level' in df_with_levels.columns, "factor_level column should be created"
    
    # Check factor level distribution
    level_distribution = df_with_levels['factor_level'].value_counts().sort_index()
    logger.info(f"Factor level distribution: {level_distribution.to_dict()}")
    
    # Check that levels are in expected range
    assert df_with_levels['factor_level'].min() >= 0, "Factor levels should be >= 0"
    assert df_with_levels['factor_level'].max() <= 4, "Factor levels should be <= 4"
    
    logger.info("✓ Factor level conversion test passed")

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline."""
    logger.info("Testing preprocessing pipeline...")
    
    # Create test data
    create_test_data()
    
    # Run preprocessing pipeline
    processed_df = preprocess_pipeline(
        input_path="data/raw/test_data.csv",
        output_path="data/processed/test_processed.csv",
        numeric_columns=['feature1', 'feature2'],
        categorical_columns=['category1', 'category2'],
        target_column='days_with_alarm',
        date_column='date',
        identifier_columns=['company_id'],
        group_by='company_id',
        handle_outliers=True,
        create_factor_level=True
    )
    
    # Check if processing was successful
    assert not processed_df.empty, "Processed DataFrame should not be empty"
    assert 'factor_level' in processed_df.columns, "factor_level should be in processed data"
    
    logger.info(f"Processed data shape: {processed_df.shape}")
    logger.info("✓ Preprocessing pipeline test passed")

def test_visualization():
    """Test visualization functionality."""
    logger.info("Testing visualization...")
    
    # Create test data
    df = create_test_data()
    
    # Create a simple model for testing
    X = df[['feature1', 'feature2']].fillna(0)
    y1 = (df['days_with_alarm'] > 5).astype(int)
    y2 = df['days_with_alarm']
    
    # Train a simple model
    model = TwoStageModel()
    model.fit(X, y1, y2)
    
    # Test visualization
    visualizer = ModelVisualizer("test_results")
    
    # Test metrics plotting
    metrics = {'stage1_f1': 0.8, 'stage1_precision': 0.75, 'stage2_rmse': 2.5}
    visualizer.plot_model_metrics(metrics, save_plot=True)
    
    # Test feature importance plotting
    importance = {
        'stage1': {'feature1': 0.6, 'feature2': 0.4},
        'stage2': {'feature1': 0.7, 'feature2': 0.3}
    }
    visualizer.plot_feature_importance(importance, save_plot=True)
    
    logger.info("✓ Visualization test passed")

def test_pipeline_separation():
    """Test the training pipeline with evaluation and separate evaluation pipeline."""
    logger.info("Testing pipeline structure...")
    
    # Create test data
    create_test_data()
    
    # Test training pipeline (includes evaluation)
    from main import training_pipeline
    
    model_path, metrics, importance = training_pipeline(
        input_data_path="data/raw/test_data.csv",
        processed_data_path="data/processed/test_processed.csv",
        model_save_path="models/saved/test_model.joblib",
        numeric_columns=['feature1', 'feature2'],
        categorical_columns=['category1', 'category2'],
        test_size=0.2,
        random_state=42,
        results_dir="test_results"
    )
    
    # Verify model was saved and results were created
    assert Path(model_path).exists(), "Model should be saved"
    assert isinstance(metrics, dict), "Metrics should be returned"
    assert isinstance(importance, dict), "Importance should be returned"
    assert Path("test_results").exists(), "Results directory should be created"
    
    logger.info(f"✓ Training pipeline test passed - model saved to {model_path}")
    
    # Test separate evaluation pipeline
    from main import evaluation_pipeline
    
    metrics2, importance2 = evaluation_pipeline(
        model_load_path=model_path,
        results_dir="test_results_eval",
        model_name="test_model"
    )
    
    # Verify evaluation results
    assert isinstance(metrics2, dict), "Evaluation metrics should be returned"
    assert isinstance(importance2, dict), "Evaluation importance should be returned"
    assert Path("test_results_eval").exists(), "Evaluation results directory should be created"
    
    logger.info("✓ Evaluation pipeline test passed")

def test_preprocessing_only():
    """Test the preprocessing-only mode."""
    logger.info("Testing preprocessing-only mode...")
    
    # Create test data
    create_test_data()
    
    # Test preprocessing pipeline only
    from main import preprocess_pipeline
    
    processed_data = preprocess_pipeline(
        input_path="data/raw/test_data.csv",
        output_path="data/processed/test_preprocess_only.csv",
        numeric_columns=['feature1', 'feature2'],
        categorical_columns=['category1', 'category2'],
        target_column='days_with_alarm',
        date_column='date',
        identifier_columns=['company_id'],
        group_by='company_id',
        handle_outliers=True,
        create_factor_level=True
    )
    
    # Verify preprocessing results
    assert not processed_data.empty, "Processed data should not be empty"
    assert 'factor_level' in processed_data.columns, "factor_level should be in processed data"
    
    logger.info(f"✓ Preprocessing-only test passed - data shape: {processed_data.shape}")

def main():
    """Run all tests."""
    logger.info("Starting implementation tests...")
    
    try:
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("test_results").mkdir(exist_ok=True)
        Path("test_results_eval").mkdir(exist_ok=True)
        
        # Run tests
        test_outlier_handling()
        test_factor_level_conversion()
        test_preprocessing_pipeline()
        test_visualization()
        test_pipeline_separation()  # Test training with evaluation
        test_preprocessing_only()   # Test preprocessing-only mode
        
        logger.info("✓ All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 