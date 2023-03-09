import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path('../../data')
df = pd.read_parquet(DATA_PATH.joinpath('processed', 'game_logs_standings_v2.parquet'))
df_test = df.sample(frac=0.2, random_state=42)
df_train = df.drop(index=df_test.index)

df_train.to_parquet(DATA_PATH.joinpath('processed', 'train.parquet'))
df_test.to_parquet(DATA_PATH.joinpath('processed', 'test.parquet'))
