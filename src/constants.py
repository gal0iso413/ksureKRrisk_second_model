"""
Constants and configuration values for the second model.
"""

# Random state for reproducibility
DEFAULT_RANDOM_STATE = 42

# Data preprocessing
DEFAULT_SCALER_TYPE = 'standard'  # 'standard' or 'minmax'
DEFAULT_ENCODING_METHOD = 'onehot'  # 'onehot' or 'label'
DEFAULT_ENCODING = 'utf-8'

# Model training
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_SPLITS = 5
DEFAULT_CV_SCORING = 'neg_mean_squared_error'
DEFAULT_CV_SHUFFLE = True

# Available model types
AVAILABLE_MODEL_TYPES = ['xgboost', 'lightgbm']

# Hyperparameter search
DEFAULT_SEARCH_METHOD = 'grid'  # 'grid' or 'random'
DEFAULT_SEARCH_CV = 3
DEFAULT_SEARCH_N_ITER = 50

# Feature engineering
DEFAULT_POLYNOMIAL_DEGREE = 2
DEFAULT_PCA_VARIANCE_THRESHOLD = 0.95

# Time series features
DEFAULT_LAG_PERIODS = [1, 2, 3]
DEFAULT_ROLLING_WINDOWS = [2, 3]

# Risk types
RISK_TYPES = {
    'normal': 0,
    'warning': 1,
    'critical': 2
}

# Model hyperparameters
XGBOOST_CLASSIFIER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'num_class': 5,
    'verbosity': 0
}

XGBOOST_REGRESSOR_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'objective': 'reg:tweedie',
    'eval_metric': 'rmsle',
    'verbosity': 0
}

LIGHTGBM_CLASSIFIER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 3,
    'verbosity': -1
}

LIGHTGBM_REGRESSOR_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'objective': 'tweedie',
    'metric': 'rmsle',
    'verbosity': -1
}

# Hyperparameter search spaces
XGBOOST_CLASSIFIER_SEARCH_SPACE = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

XGBOOST_REGRESSOR_SEARCH_SPACE = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

LIGHTGBM_CLASSIFIER_SEARCH_SPACE = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

LIGHTGBM_REGRESSOR_SEARCH_SPACE = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Korean font configuration
KOREAN_FONT_CONFIG = {
    'font_family': 'NanumGothic',
    'font_size': 12
} 