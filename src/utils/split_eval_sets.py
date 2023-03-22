import pandas as pd
import numpy as np
from pathlib import Path
from fastai.tabular.all import *

DATA_PATH = Path('../../data')
YEAR = 2009
NOT_INCLUDE_YEARS = [2020] # COVID year

df = pd.read_parquet(DATA_PATH.joinpath('processed', 'game_logs_standings.parquet'))

df = add_datepart(df, "Date", drop=False)
df = df.loc[df.Year >= YEAR, :]
df = df.loc[~df.Year.isin(NOT_INCLUDE_YEARS), :]

df_test = df.sample(frac=0.2, random_state=42)
df_train = df.drop(index=df_test.index)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

df_train.to_parquet(DATA_PATH.joinpath('processed', 'train.parquet'))
df_test.to_parquet(DATA_PATH.joinpath('processed', 'test.parquet'))
