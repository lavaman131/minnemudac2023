import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    DATA_PATH = Path("../../data")
    df_standings = pd.read_parquet(
        DATA_PATH.joinpath("processed", "All_Team_Standings_2000-2022.parquet")
    )

    df_standings["Date"] = pd.to_datetime(df_standings["Date"], format="%Y-%m-%d")

    df_game_logs = pd.read_csv(
        DATA_PATH.joinpath("raw", "game_logs.csv"), encoding="UTF-16"
    )

    df_game_logs["Date"] = pd.to_datetime(df_game_logs["Date"], format="%Y%m%d")

    merged_df = pd.merge(
        left=df_standings,
        right=df_game_logs,
        how="inner",
        on=["Date", "HomeTeam", "VisitingTeam"],
    )

    merged_df.head()

    print(df_game_logs.Date.dt.year.unique())

    merged_df.to_parquet(DATA_PATH.joinpath("processed", "game_logs_standings.parquet"))

    print(merged_df.shape)
