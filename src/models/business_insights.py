from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import List
import pandas as pd
from pathlib import Path
from src.models.features import (
    CAT_FEATURES,
    CONT_FEATURES,
    CAT_FEATURES,
    CONT_FEATURES,
)


def to_numerical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    data = np.empty((df.shape[0], len(columns)))
    for idx, col in enumerate(columns):
        mapping = {key: idx for idx, key in enumerate(np.unique(df[col]))}
        data[:, idx] = df.loc[:, col].map(mapping)
    return data


DATA_PATH = Path("/Users/alilavaee/Documents/minnemudac2023/data")
MODEL_PATH = Path("/Users/alilavaee/Documents/minnemudac2023/models")

# * NOT EMBEDDED DATA
FEATURES = CAT_FEATURES + CONT_FEATURES
df_train = pd.read_parquet(DATA_PATH.joinpath("processed", "train.parquet"))
df_val = pd.read_parquet(DATA_PATH.joinpath("processed", "val.parquet"))
df_test = pd.read_parquet(DATA_PATH.joinpath("processed", "test.parquet"))

X_train = df_train.drop(["Attendance_TRUTH_y"], axis=1)[FEATURES]
X_train[X_train.select_dtypes(['O', bool]).columns] = to_numerical(X_train, X_train.select_dtypes(['O', bool]).columns)

y_train = df_train["Attendance_TRUTH_y"]
X_val = df_val.drop(["Attendance_TRUTH_y"], axis=1)[FEATURES]
X_val[X_val.select_dtypes(['O', bool]).columns] = to_numerical(X_val, X_val.select_dtypes(['O', bool]).columns)

y_val = df_val["Attendance_TRUTH_y"]
X_test = df_test.drop(["Attendance_TRUTH_y"], axis=1)[FEATURES]
X_test[X_test.select_dtypes(['O', bool]).columns] = to_numerical(X_test, X_test.select_dtypes(['O', bool]).columns)

y_test = df_test["Attendance_TRUTH_y"]

reg = LinearRegression()
reg = reg.fit(X_train, y_train)
y_pred_linreg = reg.predict(X_test)

reg = CatBoostRegressor()
reg = reg.load_model(MODEL_PATH.joinpath("catboost.cbm"))
reg = reg.fit(X_train, y_train)
y_pred_catboost = reg.predict(X_test)

important_features = ['median_Attendance_TRUTH_y_2_yr_ago',
 'median_Attendance_TRUTH_y_1_yr_ago',
 'Dayofweek',
 'Dayofyear',
 'Year',
 'VisitingTeam',
 'HomeTeam',
 'final_HomeTeam_cLI_1_yr_ago',
 'final_VisitingTeam_cLI_1_yr_ago',
 'final_HomeTeam_W_2_yr_ago',
 'final_HomeTeam_cLI_2_yr_ago',
 'Week',
 'final_HomeTeam_W_1_yr_ago',
 'final_VisitingTeam_cLI_2_yr_ago',
 'final_VisitingTeam_W_1_yr_ago',
 'final_VisitingTeam_W_2_yr_ago',
 'final_HomeTeam_Rank_1_yr_ago',
 'HomeTeamLeague',
 'Month',
 'VisitingTeamLeague']

preds = pd.DataFrame({'lin_reg_y': y_pred_linreg, 'catboost_y': y_pred_catboost, 'y_true': y_test})
preds = pd.concat([preds, X_test[important_features]], axis=1)
preds.to_csv(DATA_PATH.joinpath("business_insights.csv"))