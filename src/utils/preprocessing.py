import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pathlib import Path
from typing import List
from fastai.tabular.all import *
import requests

# * BEST performance is 2012 for YEAR and YEARS_AGO is 3
#! note to change YEARS_AGO in features.py
YEARS_AGO = 2
YEAR = 2012
NOT_INCLUDE_YEARS = []  # COVID year
DATA_PATH = Path("/Users/alilavaee/Documents/minnemudac2023/data")

LAST_FEATURES = [
    "HomeTeam_cLI",
    "HomeTeam_Rank",
    "HomeTeam_W",
    "VisitingTeam_cLI",
    "VisitingTeam_Rank",
    "VisitingTeam_W",
]

AVG_FEATURES = ["Attendance_TRUTH_y"]

df = pd.read_parquet(DATA_PATH.joinpath("processed", f"game_logs_standings.parquet"))

df = add_datepart(df, "Date", drop=False)

df = df.loc[df.Year >= YEAR, :]
if len(NOT_INCLUDE_YEARS) > 0:
    df = df.loc[~df.Year.isin(NOT_INCLUDE_YEARS), :]


# helper function
def to_numerical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    data = np.empty((df.shape[0], len(columns)))
    for idx, col in enumerate(columns):
        mapping = {key: idx for idx, key in enumerate(np.unique(df[col]))}
        data[:, idx] = df[col].map(mapping)
    return data


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

leagues = {
    "SDP": "NL",
    "STL": "NL",
    "CHW": "AL",
    "MIL": "NL",
    "BAL": "AL",
    "TEX": "AL",
    "TOR": "AL",
    "KCR": "AL",
    "HOU": "AL",
    "ATL": "NL",
    "PIT": "NL",
    "WAS": "NL",
    "SFG": "NL",
    "MIA": "NL",
    "NYY": "AL",
    "DET": "AL",
    "COL": "NL",
    "CLE": "AL",
    "BOS": "AL",
    "CIN": "NL",
    "LAD": "NL",
    "ARI": "NL",
    "TBA": "AL",
    "SEA": "AL",
    "CHC": "NL",
    "LAA": "AL",
    "PHI": "NL",
    "OAK": "AL",
    "NYM": "NL",
    "MIN": "AL",
}

df["VisitingTeamLeague"] = df["VisitingTeam"].map(leagues)

df["HomeTeamLeague"] = df["HomeTeam"].map(leagues)

df.reset_index(inplace=True, drop=True)


# collect wins and losses for home team
df["W"] = df["W-L"].apply(lambda s: s.split("-")[0]).astype("int")
df["L"] = df["W-L"].apply(lambda s: s.split("-")[1]).astype("int")

# convert streak to quantitative
df["Streak_count"] = df["Streak"].apply(
    lambda s: s.count("+") if s.startswith("+") else -1 * s.count("-")
)

df["DayNight"] = df["Time"].apply(lambda h: pd.Timestamp(h).hour >= 7).astype("int")

# ensure appropriate data types
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start="2000-01-01", end="2022-12-31").to_pydatetime()
df["is_holiday"] = df["Date"].apply(lambda d: d in holidays).astype("int")

df["season"] = df["Month"] % 12 // 3 + 1

# df[["HomeTeam_StartingPitcher_ID", "VisitingTeam_StartingPitcher_ID"]] = to_numerical(
#     df, ["HomeTeam_StartingPitcher_ID", "VisitingTeam_StartingPitcher_ID"]
# )

df.drop(
    ["Unnamed: 3", "Unnamed: 5", "Attendance_TRUTH_x", "Attendance"],
    axis=1,
    inplace=True,
)

df = df.sort_values(by="Date", axis=0).reset_index(drop=True)


def get_previous_year_metrics(df, feature, yrs_ago=1, get_last=True):
    df_cp = df.copy()
    # Set the index of the DataFrame to be a DateTimeIndex
    group = df.groupby(["Year", "HomeTeam"])
    if get_last:
        new_col_name = f"final_{feature}_{yrs_ago}_yr_ago"
        print(f"Extracting final {feature} for {yrs_ago} yr ago")
    else:
        new_col_name = f"median_{feature}_{yrs_ago}_yr_ago"
        print(f"Extracting median {feature} for {yrs_ago} yr ago")
    if get_last:
        for idx, row in df_cp.iterrows():
            try:
                prev_metric = np.median(group.get_group((row.Year - 1, row.HomeTeam))[
                    feature
                ])
                df_cp.loc[idx, new_col_name] = prev_metric
            except:
                df_cp.loc[idx, new_col_name] = np.nan
    else:
        for idx, row in df.iterrows():
            try:
                prev_metric = group.get_group((row.Year - 1, row.HomeTeam))[
                    feature
                ].mean()
                df_cp.loc[idx, new_col_name] = prev_metric
            except:
                df_cp.loc[idx, new_col_name] = np.nan

    return df_cp


final_feature_df = []
for y in range(1, YEARS_AGO + 1):
    for feature in LAST_FEATURES:
        d = get_previous_year_metrics(
            df,
            feature,
            yrs_ago=y,
            get_last=True,
        )
        final_feature_df.append(d)

    for feature in AVG_FEATURES:
        d = get_previous_year_metrics(
            df,
            feature,
            yrs_ago=y,
            get_last=False,
        )
        final_feature_df.append(d)
df = pd.concat(final_feature_df, axis=1)

# drop any duplicate rows and columns
df = df.loc[~df.duplicated(), :].reset_index(drop=True)
df = df.loc[:, ~df.columns.duplicated()]

# "HomeTeam_cLI",
# "HomeTeam_Rank",
# "Stadium_Capacity",
# "HomeTeam_W",
# "temperature_2m_max",
# "temperature_2m_min",
# "temperature_2m_mean",
# "precipitation_sum",
# "windspeed_10m_max",
# "population",
# "avg_income",
# "lat",
# "lng",
# "windgusts_10m_max",
# "winddirection_10m_dominant",
# "HomeTeam_L",
# "HomeTeam_Streak_count",
# "HomeTeamGameNumber",
# "Rank_Diff",
# "VisitingTeam_cLI",
# "VisitingTeam_Rank",
# "VisitingTeam_W",
# "VisitingTeam_L",
# "VisitingTeam_Streak_count",
# "VisitingTeamGameNumber",


df["Rank_Diff"] = (df["HomeTeam_Rank"] - df["VisitingTeam_Rank"]).abs()

team_codes = {
    "SDP": "San Diego",
    "STL": "St. Louis",
    "CHW": "Chicago",
    "MIL": "Milwaukee",
    "BAL": "Baltimore",
    "TEX": "Texas",
    "TOR": "Toronto",
    "KCR": "Kansas City",
    "HOU": "Houston",
    "ATL": "Atlanta",
    "PIT": "Pittsburgh",
    "WAS": "Washington",
    "SFG": "San Francisco",
    "MIA": "Miami",
    "NYY": "New York",
    "DET": "Detroit",
    "COL": "Colorado",
    "CLE": "Cleveland",
    "BOS": "Boston",
    "CIN": "Cincinnati",
    "LAD": "Los Angeles",
    "ARI": "Arizona",
    "TBA": "Tampa Bay",
    "SEA": "Seattle",
    "CHC": "Chicago",
    "LAA": "Los Angeles",
    "PHI": "Philadelphia",
    "OAK": "Oakland",
    "NYM": "New York",
    "MIN": "Minnesota",
}


df["HomeTeam_City"] = df["HomeTeam"].map(team_codes)
city_coord = pd.read_csv(DATA_PATH.joinpath("raw", "city_coords.csv"), index_col=0)
df[["lat", "lng"]] = df["HomeTeam_City"].apply(lambda city: city_coord.loc[city, :])


# def get_weather(df):
#     weather_df = pd.DataFrame()
#     print("Gathering weather metrics.")
#     for (lt, lg), g in df.groupby(["lat", "lng"]):
#         g = g.sort_values(by=["Date"])
#         start_date = g["Date"].iat[0].strftime("%Y-%m-%d")
#         end_date = g["Date"].iat[-1].strftime("%Y-%m-%d")
#         url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lt}&longitude={lg}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant&timezone=America%2FNew_York&temperature_unit=fahrenheit&windspeed_unit=mph&precipitation_unit=inch"
#         payload = {}
#         headers = {}
#         response = requests.request("GET", url, headers=headers, data=payload)
#         res = response.json()
#         w = pd.DataFrame(res["daily"])
#         w["Date"] = pd.to_datetime(w["time"], format="%Y-%m-%d")
#         w = pd.merge(
#             left=w,
#             right=g[["Date", "HomeTeam", "VisitingTeam"]],
#             how="inner",
#             on="Date",
#         )
#         weather_df = pd.concat([weather_df, w], axis=0)
#     return weather_df


# weather = get_weather(df)

# print(weather.head())

# df = pd.merge(
#     left=df,
#     right=weather,
#     how="inner",
#     on=["Date", "HomeTeam"],
#     suffixes=("", "_remove"),
# ).reset_index(drop=True)

# df.drop([i for i in df.columns if "remove" in i], axis=1, inplace=True)

# print(df.head())

# load income and population data
# demographics = pd.read_excel(DATA_PATH.joinpath("processed", "City_data.xlsx"))

# demographics = demographics.rename(
#     {"year": "Year", "city_name": "HomeTeam_City"}, axis=1
# )

# df = df.merge(
#     demographics[["Year", "HomeTeam_City", "avg_income", "population"]],
#     on=["Year", "HomeTeam_City"],
#     how="inner",
# )

df[
    [
        # "StadiumID",
        "HomeTeam",
        "VisitingTeam",
        "HomeTeam_City",
        "Dayofweek",
        "VisitingTeamLeague",
        "HomeTeamLeague",
    ]
] = to_numerical(
    df,
    [
        # "BallParkID",
        "HomeTeam",
        "VisitingTeam",
        "HomeTeam_City",
        "Dayofweek",
        "VisitingTeamLeague",
        "HomeTeamLeague",
    ],
)

# drop any duplicate rows and columns
df = df.loc[~df.duplicated(), :].reset_index(drop=True)
df = df.loc[:, ~df.columns.duplicated()]
df.drop("Elapsed", axis=1, inplace=True)
df.dropna(axis=1, inplace=True, thresh=int(df.shape[0] * 0.2))
df.dropna(axis=0, inplace=True)
print(df.head())


# print(df.columns)

print(df["Attendance_TRUTH_y"].min())
print(df["Attendance_TRUTH_y"].max())


CONT_FEATURES = [
    "is_holiday",
    "Year",
    "Month",
    "Week",
    "DayNight",
    "Dayofyear",
    "season",
    "Is_month_end",
    "Is_month_start",
    "Is_quarter_end",
    "Is_quarter_start",
    "Is_year_end",
    "Is_year_start",
    # "HomeTeam_cLI",
    # "HomeTeam_Rank",
    # "Stadium_Capacity",
    # "HomeTeam_W",
    # "temperature_2m_max",
    # "temperature_2m_min",
    # "temperature_2m_mean",
    # "precipitation_sum",
    # "windspeed_10m_max",
    # "population",
    # "avg_income",
    # "lat",
    # "lng",
    # "windgusts_10m_max",
    # "winddirection_10m_dominant",
    # "HomeTeam_L",
    # "HomeTeam_Streak_count",
    # "HomeTeamGameNumber",
    # "Rank_Diff",
    # "VisitingTeam_cLI",
    # "VisitingTeam_Rank",
    # "VisitingTeam_W",
    # "VisitingTeam_L",
    # "VisitingTeam_Streak_count",
    # "VisitingTeamGameNumber",
]

historical_data = []
for y in range(1, YEARS_AGO + 1):
    for feature in AVG_FEATURES:
        historical_data.append(f"median_{feature}_{y}_yr_ago")

for y in range(1, YEARS_AGO + 1):
    for feature in LAST_FEATURES:
        historical_data.append(f"final_{feature}_{y}_yr_ago")

CONT_FEATURES.extend(historical_data)

CAT_FEATURES = [
    # "StadiumID",
    "Dayofweek",
    "VisitingTeam",
    "VisitingTeamLeague",
    "HomeTeam",
    "HomeTeamLeague",
]

print("Features:")
print(CONT_FEATURES + CAT_FEATURES)

df_test = df.sample(frac=0.2, random_state=42)
df_train = df.drop(index=df_test.index)

df_val = df_train.sample(frac=0.2, random_state=42)
df_train = df_train.drop(index=df_val.index)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

df_train.to_parquet(DATA_PATH.joinpath("processed", "train.parquet"))
df_val.to_parquet(DATA_PATH.joinpath("processed", "val.parquet"))
df_test.to_parquet(DATA_PATH.joinpath("processed", "test.parquet"))
