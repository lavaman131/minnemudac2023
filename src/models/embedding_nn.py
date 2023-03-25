from src.models.create_learner import create_model, embed_features, RMSELoss
from fastai.tabular.all import *
import torch
from src.models.features import CAT_FEATURES, CONT_FEATURES
import pandas as pd

BS = 64
RANDOM_STATE = 42
DEVICE = "mps"
DATA_PATH = Path("/Users/alilavaee/Documents/minnemudac2023/data")
MODEL_PATH = Path("/Users/alilavaee/Documents/minnemudac2023/models")

df_train = pd.read_parquet(DATA_PATH.joinpath("processed", "train.parquet"))
# TRAIN SET
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
    training=True,
)

learn.fit_one_cycle(30)
learn.save(MODEL_PATH.joinpath("embed_nn"))