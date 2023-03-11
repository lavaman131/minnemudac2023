import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pathlib import Path
from typing import List
from fastai.tabular.core import add_datepart


# helper function
def to_numerical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    data = np.empty((df.shape[0], len(columns)))
    for idx, col in enumerate(df[columns]):
        mapping = {key: idx for idx, key in enumerate(np.unique(df[col]))}
        data[:, idx] = df[col].map(mapping)
    return pd.DataFrame(data, columns=columns)


DATA_PATH = Path("../../data")

df = pd.read_parquet(DATA_PATH.joinpath("processed", "game_logs_standings.parquet"))

arena_capacity = {
    "Angel Stadium of Anaheim": 45050,
    "Rangers Ballpark in Arlington": 49115,
    "Globe Life Field in Arlington": 40738,
    "Turner Field": 49586,
    "Suntrust Park": 41149,
    "Oriole Park at Camden Yards": 45971,
    "Fenway Park": 37755,
    "Sahlen Field": 16806,
    "Wrigley Field": 41915,
    "Guaranteed Rate Field;U.S. Cellular Field": 40615,
    "Cinergy Field": 40545,
    "Great American Ballpark": 42319,
    "Progressive Field": 35041,
    "Coors Field": 50144,
    "Comerica Park": 41083,
    "TD Ballpark": 8500,
    "Field of Dreams": 8000,
    "Fort Bragg Field": 12000,
    "Minute Maid Park": 41168,
    "Kauffman Stadium": 37903,
    "The Ballpark at Disney's Wide World": 11500,
    "London Stadium": 60000,
    "Dodger Stadium": 56000,
    "Sun Life Stadium": 34742,
    "Marlins Park": 36742,
    "County Stadium": 53000,
    "Miller Park": 41828,
    "Hubert H. Humphrey Metrodome": 55333,
    "Target Field": 38949,
    "Estadio Monterrey": 27702,
    "Stade Olympique": 45623,
    "Yankee Stadium I": 56703,
    "Shea Stadium": 57000,
    "Citi Field": 41922,
    "Yankee Stadium II": 54251,
    "Oakland-Alameda County Coliseum": 46796,
    "TD Ameritrade Park": 24000,
    "Veterans Stadium": 62546,
    "Citizens Bank Park": 42793,
    "Chase Field": 48519,
    "Three Rivers Stadium": 59000,
    "PNC Park": 38362,
    "Qualcomm Stadium": 70561,
    "PETCO Park": 40845,
    "Safeco Field": 47943,
    "AT&T Park": 41915,
    "Estadio Hiram Bithorn": 19100,
    "Busch Stadium II": 50553,
    "Busch Stadium III": 43975,
    "Tropicana Field": 31042,
    "Sydney Cricket Ground": 48000,
    "Tokyo Dome": 55000,
    "Rogers Centre": 53340,
    "Robert F. Kennedy Stadium": 56638,
    "Nationals Park": 41168,
    "BB&T Ballpark at Bowman Field": 2300,
}

stadiums = pd.read_csv(DATA_PATH.joinpath("raw", "stadiums.csv"))
stadiums = stadiums.loc[
    stadiums.PARKID.isin(df["BallParkID"]), ["PARKID", "NAME"]
].reset_index(drop=True)

stadiums["capacity"] = stadiums.NAME.map(arena_capacity)

mapping = dict(zip(stadiums.PARKID, stadiums.capacity))
df["Stadium_Capacity"] = df["BallParkID"].map(mapping)

# drops too many cols
# df['GB'] = pd.to_numeric(df['GB'], errors='coerce')
df.drop("GB", axis="columns", inplace=True)

rev_mapping = {
    "SDN": "SDP",
    "SLN": "STL",
    "CHA": "CHW",
    "MIL": "MIL",
    "BAL": "BAL",
    "TEX": "TEX",
    "TOR": "TOR",
    "KCA": "KCR",
    "HOU": "HOU",
    "ATL": "ATL",
    "PIT": "PIT",
    "MON": "WAS",
    "WAS": "WAS",
    "SFN": "SFG",
    "FLO": "MIA",
    "MIA": "MIA",
    "NYA": "NYY",
    "DET": "DET",
    "COL": "COL",
    "CLE": "CLE",
    "BOS": "BOS",
    "CIN": "CIN",
    "LAN": "LAD",
    "ARI": "ARI",
    "TBA": "TBA",
    "SEA": "SEA",
    "CHN": "CHC",
    "ANA": "LAA",
    "PHI": "PHI",
    "OAK": "OAK",
    "NYN": "NYM",
    "MIN": "MIN",
}

df["Tm"] = df.Tm.map(rev_mapping)
df["HomeTeam"] = df.HomeTeam.map(rev_mapping)
df["VisitingTeam"] = df.VisitingTeam.map(rev_mapping)

# collect wins and losses for home team
df["W"] = df["W-L"].apply(lambda s: s.split("-")[0]).astype("int")
df["L"] = df["W-L"].apply(lambda s: s.split("-")[1]).astype("int")

# convert streak to quantitative
df["Streak_count"] = df["Streak"].apply(
    lambda s: s.count("+") if s.startswith("+") else -1 * s.count("-")
)

df[["DayNight", "StadiumID", "VisitingTeamLeague", "HomeTeamLeague"]] = to_numerical(
    df, ["DayNight", "BallParkID", "VisitingTeamLeague", "HomeTeamLeague"]
)

# df = pd.concat([df, pd.get_dummies(df["DayofWeek"])], axis=1)

# df["Year"] = df["Date"].apply(lambda d: d.year)
# df["Month"] = df["Date"].apply(lambda d: d.month)
# df["Day"] = df["Date"].apply(lambda d: d.day)

# df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M").apply(lambda d: d.hour)
# df["Minute"] = pd.to_datetime(df["Time"], format="%H:%M").apply(lambda d: d.minute)

# ensure appropriate data types
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start="2000-01-01", end="2022-12-31").to_pydatetime()
df["is_holiday"] = df["Date"].apply(lambda d: d in holidays).astype("int")

df = add_datepart(df, "Date", drop=False)

# df["season"] = df["Month"] % 12 // 3 + 1

df[["HomeTeam_StartingPitcher_ID", "VisitingTeam_StartingPitcher_ID"]] = to_numerical(
    df, ["HomeTeam_StartingPitcher_ID", "VisitingTeam_StartingPitcher_ID"]
)


group = df.groupby(["Year", "HomeTeam", "VisitingTeam"])["Attendance_TRUTH_y"].mean()
years_ago = 3
for idx, row in df.iterrows():
    # year is at least 2001
    for y in range(1, years_ago + 1):
        if row.Year > (df.Year.min() + y - 1):
            try:
                # find average attendance of all previous year matchups of home team with visting team
                df.loc[idx, f"avg_attendance_{y}_yr_ago"] = group[row.Year - y][
                    row.HomeTeam
                ][row.VisitingTeam]
            except KeyError:
                # find average attendance of all previous year games of home team
                df.loc[idx, f"avg_attendance_{y}_yr_ago"] = df.loc[
                    df.Year == row.Year - y, "Attendance_TRUTH_y"
                ].mean()
        else:
            # year is 2000 and we don't have any information
            df.loc[idx, f"avg_attendance_{y}_yr_ago"] = np.nan


# final cleaning
df = df.loc[df.Tm == df.HomeTeam]
df = df[df.columns.unique()]
# filter by threshold of not null values
thresh = 0.8
df = df[
    [col for col in df.columns if df[col].notnull().sum() > df[col].shape[0] * thresh]
]
# drop null rows
df.dropna(axis=0, inplace=True)
df.drop(["Unnamed: 3", "Attendance_TRUTH_x", "Attendance"], axis=1, inplace=True)


# save processed data
df.to_parquet(DATA_PATH.joinpath("processed", "game_logs_standings_v2.parquet"))
