from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from typing import List
from src.models.features import CAT_FEATURES, CONT_FEATURES

def to_numerical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    data = np.empty((df.shape[0], len(columns)))
    for idx, col in enumerate(columns):
        mapping = {key: idx for idx, key in enumerate(np.unique(df[col]))}
        data[:, idx] = df.loc[:, col].map(mapping)
    return data

MODEL_PATH = Path("/Users/alilavaee/Documents/minnemudac2023/models")
DATA_PATH = Path("/Users/alilavaee/Documents/minnemudac2023/data")

FEATURES = CAT_FEATURES + CONT_FEATURES

df = pd.read_csv(DATA_PATH.joinpath("prediction", "forecast_data.csv"), index_col=0)
X_test = df[FEATURES]

X_test[X_test.select_dtypes(['O', bool]).columns] = to_numerical(X_test, X_test.select_dtypes(['O', bool]).columns)

print(X_test.dtypes)
reg = CatBoostRegressor()
reg = reg.load_model(MODEL_PATH.joinpath("catboost.cbm"))

df["minnemudac_teamid"] = "U03"
df["predicted_attendance"] = reg.predict(X_test)

print(df[["minnemudac_teamid", "predicted_attendance"]])

submission = pd.read_csv(DATA_PATH.joinpath("prediction", "submission.csv"))

submission.dropna(axis=1, inplace=True)

final_submission = pd.concat([submission, df[["minnemudac_teamid", "predicted_attendance"]]], axis=1)

print(final_submission.head())
final_submission.to_csv(DATA_PATH.joinpath("prediction", "2023_MLBSchedule_submission.csv"))

