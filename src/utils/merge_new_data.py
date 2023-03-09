import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path('../../data')
df_standings = pd.read_parquet(DATA_PATH.joinpath('processed',
                                                  'All_Team_Standings_2000-2022.parquet'))
df_game_logs = pd.read_parquet(DATA_PATH.joinpath('processed', 'game_logs.parquet'))

target_columns = ['Gm#', 'Date', 'Tm', 'HomeTeam', 'VisitingTeam', 'W/L', 'R', 'RA', 'W-L',
                  'Rank', 'GB', 'Time', 'cLI', 'Streak']

df_standings = df_standings[target_columns]

merged_df = pd.merge(left=df_standings, right=df_game_logs,
                     how='inner', on=['Date', 'HomeTeam', 'VisitingTeam']).dropna() \
                                                            .reset_index(drop=True)

merged_df.to_parquet(DATA_PATH.joinpath('processed', 'game_logs_standings.parquet'))

print(merged_df.shape)

