import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from features import FEATURES, DT_PARAMS


DATA_PATH = Path('../../data')
df = pd.read_parquet(DATA_PATH.joinpath('processed', 'train.parquet'))

X = df[FEATURES].to_numpy()
y = df['Attendance'].to_numpy()

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# params =  {
#     'max_leaf_nodes': np.arange(2, 20),
#     'max_depth': np.arange(1, 20)
# }

# reg = DecisionTreeRegressor(criterion='friedman_mse')

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

for i, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    reg = DecisionTreeRegressor(**DT_PARAMS, criterion='friedman_mse', random_state=42)
    X_train, y_train = X[train_index], y[train_index]
    reg = reg.fit(X_train, y_train)
    y_pred_train = reg.predict(X_train)
    MAE_train = np.abs(y_pred_train - y_train).mean()
    X_val, y_val = X[val_index], y[val_index]
    y_pred_val = reg.predict(X_val)
    MAE_val = np.abs(y_pred_val - y_val).mean()
    print(f"  Train MAE: {MAE_train}")
    print(f"  Validation MAE: {MAE_val}")