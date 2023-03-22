import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from catboost import CatBoostRegressor, EFstrType, Pool
import sys
from src.models.features import FEATURES, CB_PARAMS
from sklearn.metrics import r2_score


DATA_PATH = Path("../../data")
df = pd.read_parquet(DATA_PATH.joinpath("processed", "train_embed.parquet"))

df_X = df.drop(["Attendance_TRUTH_y"], axis=1)
df_y = df["Attendance_TRUTH_y"]

X = df_X.to_numpy()
y = df_y.to_numpy()

print("5-Fold Cross Validation Results:")
kf = KFold(n_splits=5, random_state=42, shuffle=True)

for i, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    X_train, y_train = X[train_index], y[train_index]
    reg = CatBoostRegressor(**CB_PARAMS, loss_function="RMSE", random_seed=42)
    reg = reg.fit(X_train, y_train, verbose=False)
    print(
        reg.get_feature_importance(
            data=Pool(data=df_X.iloc[train_index], label=df_y.iloc[train_index]),
            reference_data=None,
            type=EFstrType.FeatureImportance,
            prettified=True,
        ).head(20)
    )
    y_pred_train = reg.predict(X_train)
    MAE_train = np.abs(y_pred_train - y_train).mean()
    X_val, y_val = X[val_index], y[val_index]
    y_pred_val = reg.predict(X_val)
    r_2 = r2_score(y_val, y_pred_val)
    print(f"R^2 {r_2}")
    MAE_val = np.abs(y_pred_val - y_val).mean()
    print(f"  Train MAE: {MAE_train}")
    print(f"  Validation MAE: {MAE_val}")
