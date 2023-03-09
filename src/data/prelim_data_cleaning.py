import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path('../../data')

df = pd.read_csv(DATA_PATH.joinpath('raw', 'game_logs.csv'), encoding='UTF-16')
# filter by threshold of not null values
thresh = 0.8
df = df[[col for col in df.columns if df[col].notnull().sum() > df[col].shape[0] * thresh]]
# ensure appropriate data types
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
# drop null rows
df.dropna(axis=0, inplace=True)
# export to parquet
df.to_parquet(DATA_PATH.joinpath('processed', 'game_logs.parquet'))