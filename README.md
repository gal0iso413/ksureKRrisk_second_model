# Second Model: Two-Stage Risk Prediction

This model implements a two-stage approach for predicting risk types and their durations:

1. First Stage: Classification model to predict risk type
2. Second Stage: Regression model to predict duration for non-normal cases

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
   - Stage 1: Predicts risk type (classification)
   - Stage 2: Predicts duration for non-normal cases (regression)

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

6. **Factor Level Conversion**
   - Converts factor codes (`발생사유항목코드_1`, `발생사유항목코드_2`, etc.) to factor levels (0-4)
   - Based on count of non-null factor codes
   - Mapping: 0=no codes, 1=1-2 codes, 2=3-4 codes, 3=5-6 codes, 4=7+ codes

7. **Comprehensive Visualization Pipeline**
   - Model metrics plots for both stages
   - Feature importance plots
   - SHAP analysis plots for model interpretability
   - Prediction vs actual plots
   - Confusion matrix for classification

8. **SHAP Analysis Integration**
   - SHAP summary plots for both model stages
   - Focus on stage 2 (regression) as requested
   - Automatic sampling for large datasets
   - Support for XGBoost and LightGBM models

9. **Automatic Result Management**
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
    target_column='duration',
    date_column='date',
    identifier_columns=['company_id'],
    group_by=['company_id', 'region'],
    handle_outliers=True,      # Enable outlier handling
    create_factor_level=True   # Enable factor level conversion
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

#### Factor Level Conversion
```python
from src.data_preprocessing import FactorLevelConverter

df_with_levels = FactorLevelConverter.convert_to_factor_level(
    df=df,
    factor_code_prefix='발생사유항목코드_',
    output_column='factor_level'
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
    y1=risk_types,
    y2=durations,
    feature_names=feature_names
)

# Make predictions
risk_types, durations = model.predict(X_test)

# Evaluate model
metrics = model.evaluate(X_test, y1_test, y2_test)
```

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
```

## Configuration

Key configuration parameters can be found in `src/constants.py`:

- Model hyperparameters
- Preprocessing settings
- Feature engineering parameters
- Risk types

### New Configuration Options

#### Factor Level Mapping
Adjust the factor level mapping in `src/data_preprocessing.py`:

```python
def map_to_factor_level(count):
    if count == 0:
        return 0
    elif count <= 2:
        return 1
    elif count <= 4:
        return 2
    elif count <= 6:
        return 3
    else:
        return 4
```

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
- Factor level conversion
- Complete preprocessing pipeline
- Visualization generation

## Handling Zero-Inflated Data

The model handles zero-inflated data through:

1. Two-stage approach:
   - First stage identifies if there will be a risk
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

- Graceful handling of missing factor codes
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
