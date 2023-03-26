import pandas as pd
import numpy as np
from pathlib import Path
from src.models.features import (
    CAT_FEATURES,
    CONT_FEATURES,
    CB_PARAMS,
    DT_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
    CAT_FEATURES,
    CONT_FEATURES,
)

# import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# import models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from fastai.tabular.all import *
from src.models.create_learner import create_model, RMSELoss

REPORTS = Path("/Users/alilavaee/Documents/minnemudac2023/reports")

def data_to_numeric(df):
    df_cp = df.copy()
    cols = [
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ]

    for col in df_cp.columns:
        if col in cols:
            df_cp[col] = df_cp[col].astype(int)
    return df_cp
    # return pd.concat([pd.to_numeric(df[col]) for col in df.columns], axis=1)


MODEL_PATH = Path("/Users/alilavaee/Documents/minnemudac2023/models")
RANDOM_STATE = 42
BS = 64
DEVICE = "mps"
EVAL_TEST = True


DATA_PATH = Path("/Users/alilavaee/Documents/minnemudac2023/data")
MODEL_PATH = Path("/Users/alilavaee/Documents/minnemudac2023/models")


# * NOT EMBEDDED DATA
FEATURES = CAT_FEATURES + CONT_FEATURES
df_train = pd.read_parquet(DATA_PATH.joinpath("processed", "train.parquet"))
df_val = pd.read_parquet(DATA_PATH.joinpath("processed", "val.parquet"))
df_test = pd.read_parquet(DATA_PATH.joinpath("processed", "test.parquet"))

X_train = data_to_numeric(df_train.drop(["Attendance_TRUTH_y"], axis=1)[FEATURES])
y_train = df_train["Attendance_TRUTH_y"]
X_val = data_to_numeric(df_val.drop(["Attendance_TRUTH_y"], axis=1)[FEATURES])
y_val = df_val["Attendance_TRUTH_y"]
X_test = data_to_numeric(df_test.drop(["Attendance_TRUTH_y"], axis=1)[FEATURES])
y_test = df_test["Attendance_TRUTH_y"]

sets = [(X_train, y_train), (X_val, y_val)]

if EVAL_TEST:
    sets.append((X_test, y_test))

# ? EMBEDDED DATA
# df_train_embed = pd.read_parquet(DATA_PATH.joinpath("processed", "train_embed.parquet"))
# df_val_embed = pd.read_parquet(DATA_PATH.joinpath("processed", "val_embed.parquet"))
# df_test_embed = pd.read_parquet(DATA_PATH.joinpath("processed", "test_embed.parquet"))

# X_train_embed = df_train_embed.drop(["Attendance_TRUTH_y"], axis=1)
# y_train_embed = df_train_embed["Attendance_TRUTH_y"]
# X_val_embed = df_val_embed.drop(["Attendance_TRUTH_y"], axis=1)
# y_val_embed = df_val_embed["Attendance_TRUTH_y"]
# X_test_embed = df_test_embed.drop(["Attendance_TRUTH_y"], axis=1)
# y_test_embed = df_test_embed["Attendance_TRUTH_y"]


splits_train = RandomSplitter(valid_pct=0.2, seed=RANDOM_STATE)(range_of(df_train))
to_train = TabularPandas(
    df_train,
    procs=[Categorify, FillMissing, Normalize],
    cat_names=CAT_FEATURES,
    cont_names=CONT_FEATURES,
    y_names="Attendance_TRUTH_y",
    splits=splits_train,
    device=DEVICE,
)
dls_train = to_train.dataloaders(bs=BS)

learn = create_model(
    dls=dls_train,
    metrics=mae,
    loss_func=None,
    cat_names=CAT_FEATURES,
    cont_names=CONT_FEATURES,
    training=False,
    save_model_path=MODEL_PATH.joinpath("embed_nn"),
)


model_performances = {
    "Model_Name": (["Embedding NN"] * len(sets))
    + (["Linear Regression"] * len(sets))
    + (["Decision Tree Regressor"] * len(sets))
    + (["Random Forest Regressor"] * len(sets))
    + (["XGBoost Regressor"] * len(sets))
    + (["CatBoost Regressor"] * len(sets)),
    "Set": [],
    "RMSE": [],
    "MAE": [],
}

set_id = {0: "Train", 1: "Validation", 2: "Test"}

DL_MODELS = [learn]

print("DEEP LEARNING MODEL PERFORMANCE")
for model in DL_MODELS:
    for idx, (X, y_true) in enumerate(sets):
        dl = model.dls.test_dl(X)
        dl.dataset.conts = dl.dataset.conts.astype(np.float32)
        inp, y_pred, _, dec_preds = model.get_preds(
            dl=dl, with_input=True, with_decoded=True
        )
        MAE = mean_absolute_error(y_true, y_pred)
        RMSE = mean_squared_error(y_true, y_pred, squared=False)
        model_performances["MAE"].append(MAE)
        model_performances["RMSE"].append(RMSE)
        model_performances["Set"].append(set_id[idx])

        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")
    print()

print("MACHINE LEARNING MODEL PERFORMANCE")
ML_MODELS = [
    LinearRegression(),
    DecisionTreeRegressor(criterion="friedman_mse", **DT_PARAMS),
    RandomForestRegressor(
        criterion="friedman_mse", random_state=RANDOM_STATE, **RF_PARAMS
    ),
    xgb.XGBRegressor(
        tree_method="hist",
        eval_metric=mean_absolute_error,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **XGB_PARAMS,
    ),
    CatBoostRegressor(loss_function="RMSE", random_seed=RANDOM_STATE, **CB_PARAMS),
]

for model in ML_MODELS:
    if not isinstance(model, CatBoostRegressor):
        model = model.fit(X_train, y_train)
    elif isinstance(model, CatBoostRegressor):
        model = model.fit(X_train, y_train, verbose=False)
        # model.save_model(MODEL_PATH.joinpath("catboost.cbm"))
    elif isinstance(model, LinearRegression):
        model = model.fit(X_train[CONT_FEATURES], y_train)

    for idx, (X, y_true) in enumerate(sets):
        if isinstance(model, (LinearRegression, CatBoostRegressor)):
            y_pred = model.predict(X)
        elif isinstance(model, LinearRegression):
            y_pred = model.predict(X[CONT_FEATURES])
        else:
            y_pred = model.predict(X)
        MAE = mean_absolute_error(y_true, y_pred)
        RMSE = mean_squared_error(y_true, y_pred, squared=False)
        print(f"MAPE {(np.abs(y_true - y_pred) / y_true).mean()}")
        model_performances["MAE"].append(MAE)
        model_performances["RMSE"].append(RMSE)
        model_performances["Set"].append(set_id[idx])

        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")
    print()

# model_performances = pd.DataFrame(model_performances, index=np.arange(len(model_performances["Model_Name"])))

# model_performances.to_csv(REPORTS.joinpath("model_performances.csv"))