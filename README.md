# Second Model: Two-Stage Risk Prediction

This model implements a two-stage approach for predicting alarm types and their durations:

1. First Stage: Classification model to predict alarm type
2. Second Stage: Regression model to predict alarm duration for non-normal cases

## Project Structure

```
second_model/
├── models/
│   ├── __init__.py
│   └── two_stage_model.py      # Main model implementation
├── src/
│   ├── __init__.py
│   ├── constants.py           # Configuration and constants
│   ├── data_preprocessing.py  # Data preprocessing utilities
│   ├── visualization.py       # Visualization and SHAP analysis
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py  # Logging configuration
│       └── common.py          # Common utilities
├── results/                   # Auto-created results directory
│   ├── plots/                 # Model metrics and prediction plots
│   ├── shap/                  # SHAP analysis plots
│   └── feature_importance/    # Feature importance plots
├── test_implementation.py     # Test script for new features
└── README.md
```

## Key Features

### Core Features
1. **Two-Stage Modeling**
   - Stage 1: Predicts alarm type (classification)
   - Stage 2: Predicts alarm duration for non-normal cases (regression)

2. **Data Preprocessing**
   - Regression-specific preprocessing
   - Zero-inflated data handling
   - Time series feature engineering
   - Lag and rolling statistics

3. **Model Support**
   - XGBoost
   - LightGBM
   - Configurable hyperparameters

4. **Feature Engineering**
   - Time-based features
   - Lag features
   - Rolling statistics
   - Year-over-year changes

### 🆕 New Features (Latest Update)

5. **Outlier Handling (IQR Method)**
   - Automatic outlier detection and capping
   - Configurable IQR threshold (default: 1.5)
   - Applied to numeric columns during preprocessing

6. **Comprehensive Visualization Pipeline**
   - Model metrics plots for both stages
   - Feature importance plots
   - SHAP analysis plots for model interpretability
   - Prediction vs actual plots
   - Confusion matrix for classification

7. **SHAP Analysis Integration**
   - SHAP summary plots for both model stages
   - Focus on stage 2 (regression) as requested
   - Automatic sampling for large datasets
   - Support for XGBoost and LightGBM models

8. **Automatic Result Management**
   - Auto-creates organized result folders
   - Saves all plots and analysis results
   - Structured output organization

## Usage

### 1. Training Pipeline (Recommended)

```bash
# Train model with evaluation and visualization (default)
python main.py --mode train

# Train with custom paths
python main.py --mode train --input-data data/raw/my_data.csv --model-path models/my_model.joblib --results-dir my_results
```

### 2. Evaluation Pipeline (Existing Models)

```bash
# Evaluate existing trained model
python main.py --mode evaluate

# Evaluate with custom paths
python main.py --mode evaluate --model-path models/my_model.joblib --results-dir my_results
```

### 3. Preprocessing Only

```bash
# Run preprocessing pipeline only (original functionality)
python main.py --mode preprocess

# Preprocess with custom input
python main.py --mode preprocess --input-data data/raw/my_data.csv
```

### 4. Separate Pipeline Functions

#### Training Pipeline (with evaluation)
```python
from main import training_pipeline

# Train, evaluate, and create visualizations
model_path, metrics, importance = training_pipeline(
    input_data_path='data/raw/input_data.csv',
    processed_data_path='data/processed/processed_data.csv',
    model_save_path='models/saved/two_stage_model.joblib',
    numeric_columns=['feature1', 'feature2'],
    categorical_columns=['category1', 'category2'],
    test_size=0.2,
    random_state=42,
    results_dir='results'
)
```

#### Evaluation Pipeline
```python
from main import evaluation_pipeline

# Load model and create visualizations
metrics, importance = evaluation_pipeline(
    model_load_path='models/saved/two_stage_model.joblib',
    results_dir='results',
    model_name='two_stage_model'
)
```

### 5. Data Preprocessing with New Features

```python
from src.data_preprocessing import preprocess_pipeline

processed_data = preprocess_pipeline(
    input_path='data/raw/input_data.csv',
    output_path='data/processed/processed_data.csv',
    numeric_columns=['feature1', 'feature2'],
    categorical_columns=['category1', 'category2'],
    target_column='alarm_day',
    date_column='date',
    identifier_columns=['company_id'],
    group_by=['company_id', 'region'],
    handle_outliers=True      # Enable outlier handling
)
```

### 6. Individual Feature Usage

#### Outlier Handling
```python
from src.data_preprocessing import OutlierHandler

df_cleaned = OutlierHandler.handle_outliers_iqr(
    df=df,
    columns=['feature1', 'feature2'],
    threshold=1.5
)
```

#### Visualization
```python
from src.visualization import create_visualization_pipeline

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
```

### 7. Model Training

```python
from models.two_stage_model import TwoStageModel

# Initialize model
model = TwoStageModel(
    stage1_model_type='xgboost',
    stage2_model_type='xgboost'
)

# Train model
model.fit(
    X=features,
    y1=alarm_types,      # Stage 1: Classification target
    y2=alarm_days,       # Stage 2: Regression target
    feature_names=feature_names
)

# Make predictions
alarm_types, alarm_days = model.predict(X_test)

# Evaluate model
metrics = model.evaluate(X_test, y1_test, y2_test)
```

## Data Requirements

### Required Columns
Your input data should include:
- `date`: Date column for time series features
- `company_id`: Company identifier for grouping
- `alarm_type`: Stage 1 target (classification) - e.g., 'normal', 'warning', 'critical'
- `alarm_day`: Stage 2 target (regression) - number of days with alarm

### Data Preprocessing
- Factor level conversion should be done outside the ML pipeline
- Provide clean, preprocessed data as input
- The pipeline focuses on ML-specific preprocessing (outliers, scaling, encoding)

## Dependencies

### Core Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
python-dateutil>=2.8.2
pytz>=2021.3
typing-extensions>=4.0.0
```

### 🆕 New Dependencies (Latest Update)
```
shap==0.46.0
matplotlib==3.10.0
matplotlib-inline==0.1.6
seaborn==0.13.2
joblib>=1.1.0
```

## Configuration

Key configuration parameters can be found in `src/constants.py`:

- Model hyperparameters
- Preprocessing settings
- Feature engineering parameters
- Risk types

### New Configuration Options

#### Outlier Threshold
Adjust the IQR threshold in `OutlierHandler.handle_outliers_iqr()` (default: 1.5).

#### SHAP Sample Size
Adjust the SHAP analysis sample size in `ModelVisualizer.plot_shap_analysis()` (default: 100).

## Output Files

When running the complete pipeline, the following files will be generated:

### Plots (`results/plots/`)
- `two_stage_model_metrics.png` - Model performance metrics
- `two_stage_model_stage1_confusion_matrix.png` - Classification confusion matrix
- `two_stage_model_stage2_pred_vs_actual.png` - Regression prediction vs actual

### SHAP Analysis (`results/shap/`)
- `two_stage_model_stage1_shap_summary.png` - Stage 1 SHAP summary
- `two_stage_model_stage2_shap_summary.png` - Stage 2 SHAP summary

### Feature Importance (`results/feature_importance/`)
- `two_stage_model_feature_importance.png` - Feature importance for both stages

## Testing

Test the new features with the provided test script:

```bash
python test_implementation.py
```

This script tests:
- Outlier handling functionality
- Complete preprocessing pipeline
- Visualization generation

## Handling Zero-Inflated Data

The model handles zero-inflated data through:

1. Two-stage approach:
   - First stage identifies if there will be an alarm
   - Second stage predicts duration only for non-normal cases

2. Special preprocessing:
   - Logging of zero/non-zero distribution
   - Statistics for non-zero values
   - Appropriate scaling and transformation

## Time Series Features

Even with limited yearly data (2021-2024), the model creates useful time features:

1. Basic time features:
   - Year
   - Quarter
   - Month
   - Day of week

2. Derived features:
   - Year-over-year changes
   - Lag features
   - Rolling statistics

**Note**: Time series features are only created when sufficient historical data is available.

## Model Evaluation

The model provides comprehensive evaluation metrics:

1. Classification metrics (Stage 1):
   - F1 Score
   - Precision
   - Recall

2. Regression metrics (Stage 2):
   - RMSE
   - MAE

3. **🆕 New Evaluation Features**:
   - SHAP analysis for model interpretability
   - Feature importance visualization
   - Prediction vs actual plots
   - Confusion matrix visualization

## Error Handling

All new features include comprehensive error handling:

- Graceful handling of missing data
- Automatic skipping of time series features when insufficient data
- Robust outlier detection and handling
- Comprehensive logging throughout the pipeline

## Future Improvements

1. Add more model types
2. Implement ensemble methods
3. Add more sophisticated time series features
4. Improve zero-inflated data handling
5. Add more SHAP plot types
6. Implement cross-validation for hyperparameter tuning
