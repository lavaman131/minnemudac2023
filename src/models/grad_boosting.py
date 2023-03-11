import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from src.models.features import FEATURES, XGB_PARAMS


DATA_PATH = Path('../../data')
df = pd.read_parquet(DATA_PATH.joinpath('processed', 'train.parquet'))

X = df.drop(['Attendance_TRUTH_y'], axis=1).to_numpy()
y = df['Attendance_TRUTH_y'].to_numpy()

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# params =  {
#     'n_estimators': np.arange(750, 1050, 50),
#     # 'learning_rate': np.arange(0.1, 1.05, 0.05)
# }

# reg = xgb.XGBRegressor(
#         **XGB_PARAMS,
#         tree_method="hist",
#         eval_metric=mean_absolute_error,
#         random_state=42,
#         n_jobs=-1,
#     )

# print('Running grid search for best parameters:')
# # Create gridsearch instance
# grid = GridSearchCV(estimator=reg,
#                     param_grid=params,
#                     scoring='neg_mean_absolute_error', # use MAE
#                     cv=10,
#                     n_jobs=-1,
#                     verbose=0)

# # Fit the model
# grid.fit(X_train, y_train)

# # Assess the score
# print(grid.best_score_, grid.best_params_)

print('5-Fold Cross Validation Results:')
kf = KFold(n_splits=5, random_state=42, shuffle=True)

# XGB_PARAMS.update(grid.best_params_)

for i, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    reg = xgb.XGBRegressor(
        **XGB_PARAMS,
        tree_method="hist",
        eval_metric=mean_absolute_error,
        random_state=42,
    )
    X_train, y_train = X[train_index], y[train_index]
    reg = reg.fit(X_train, y_train)
    y_pred_train = reg.predict(X_train)
    MAE_train = np.abs(y_pred_train - y_train).mean()
    X_val, y_val = X[val_index], y[val_index]
    y_pred_val = reg.predict(X_val)
    MAE_val = np.abs(y_pred_val - y_val).mean()
    print(f"  Train MAE: {MAE_train}")
    print(f"  Validation MAE: {MAE_val}")
    
    