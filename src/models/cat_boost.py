import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from catboost import CatBoostRegressor
import sys
from src.models.features import FEATURES, CATB_PARAMS


DATA_PATH = Path('../../data')
df = pd.read_parquet(DATA_PATH.joinpath('processed', 'train.parquet'))

X = df[FEATURES].to_numpy()
y = df['Attendance_TRUTH_y'].to_numpy()

# params =  {
#     'n_estimators': np.arange(5, 105, 5),
#     'depth': np.arange(1, 6),
#     'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1],
#     'l2_leaf_reg': [1, 3, 5, 7, 9]
# }

# randomized_search_result = reg.randomized_search(params,
#                                                  X=X,
#                                                  y=y,
#                                                  cv=5,
#                                                  n_iter=20,
#                                                  partition_random_seed=42,
#                                                  calc_cv_statistics=True,
#                                                  search_by_train_test_split=True,
#                                                  refit=True,
#                                                  shuffle=True,
#                                                  stratified=False,
#                                                  plot=False,
#                                                  train_size=0.8,
#                                                  verbose=False,
#                                                  log_cout=sys.stdout,
#                                                  log_cerr=sys.stderr)

print('5-Fold Cross Validation Results:')
kf = KFold(n_splits=5, random_state=42, shuffle=True)

for i, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    X_train, y_train = X[train_index], y[train_index]
    reg = CatBoostRegressor(loss_function='RMSE', random_seed=42)
    reg = reg.fit(X_train, y_train, verbose=False)
    y_pred_train = reg.predict(X_train)
    MAE_train = np.abs(y_pred_train - y_train).mean()
    X_val, y_val = X[val_index], y[val_index]
    y_pred_val = reg.predict(X_val)
    MAE_val = np.abs(y_pred_val - y_val).mean()
    print(f"  Train MAE: {MAE_train}")
    print(f"  Validation MAE: {MAE_val}")