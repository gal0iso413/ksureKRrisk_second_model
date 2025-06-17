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
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py  # Logging configuration
│       └── common.py          # Common utilities
└── README.md
```

## Key Features

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

## Usage

### 1. Data Preprocessing

```python
from src.data_preprocessing import preprocess_pipeline

preprocess_pipeline(
    input_path='data.csv',
    output_path='processed_data.csv',
    numeric_columns=['feature1', 'feature2'],
    categorical_columns=['category1', 'category2'],
    target_column='duration',
    date_column='date',
    identifier_columns=['company_id'],
    group_by='company_id'
)
```

### 2. Model Training

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

## Configuration

Key configuration parameters can be found in `src/constants.py`:

- Model hyperparameters
- Preprocessing settings
- Feature engineering parameters
- Risk types

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

## Model Evaluation

The model provides comprehensive evaluation metrics:

1. Classification metrics (Stage 1):
   - F1 Score
   - Precision
   - Recall

2. Regression metrics (Stage 2):
   - RMSE
   - MAE

## Future Improvements

1. Add more model types
2. Implement ensemble methods
3. Add more sophisticated time series features
4. Improve zero-inflated data handling
5. Add model interpretation tools 