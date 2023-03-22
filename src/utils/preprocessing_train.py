import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pathlib import Path
from typing import List
from tqdm import tqdm
from fastai.tabular.all import *
from torch import nn
import requests
from dotenv import dotenv_values
import json
import googlemaps
import datetime

config = dotenv_values("../../.env")
API_KEY = config["MAPS_API_KEY"]
YEARS_AGO = 1
FEATURES = ["HomeTeam_Rank", "HomeTeam_W"]

gmaps = googlemaps.Client(key=API_KEY)


# helper function
def to_numerical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    data = np.empty((df.shape[0], len(columns)))
    for idx, col in enumerate(columns):
        mapping = {key: idx for idx, key in enumerate(np.unique(df[col]))}
        data[:, idx] = df[col].map(mapping)
    return pd.DataFrame(data, columns=columns)


DATA_PATH = Path("../../data")

df = pd.read_parquet(DATA_PATH.joinpath("processed", "train.parquet"))

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

df["season"] = df["Month"] % 12 // 3 + 1

df[["HomeTeam_StartingPitcher_ID", "VisitingTeam_StartingPitcher_ID"]] = to_numerical(
    df, ["HomeTeam_StartingPitcher_ID", "VisitingTeam_StartingPitcher_ID"]
)

df.drop(
    ["Unnamed: 3", "Unnamed: 5", "Attendance_TRUTH_x", "Attendance"],
    axis=1,
    inplace=True,
)


# get previous year metrics
def get_previous_year_metrics(df, feature, yrs_ago=1):
    group = df.groupby(["Year", "HomeTeam"])
    print(f"Extracting final {feature} for {yrs_ago} yr ago")
    new_col_name = f"final_{feature}_{yrs_ago}_yr_ago"
    last_feature = group[feature].transform(lambda x: x.max(skipna=True))
    filtered_group = group.filter(
        lambda x: df.Year.min() + yrs_ago <= x["Year"].max() <= df.Year.max()
    )
    filtered_group[new_col_name] = last_feature.shift(yrs_ago)
    return filtered_group


def get_avg_attendance(df, yrs_ago=1):
    group = df.groupby(["Year", "HomeTeam", "VisitingTeam"])
    print(f"Extracting average attendance for {yrs_ago} yr ago")
    new_col_name = f"avg_attendance_{yrs_ago}_yr_ago"
    avg_attendance = group["Attendance_TRUTH_y"].transform(
        lambda x: x.mean(skipna=True)
    )
    filtered_group = group.filter(
        lambda x: df.Year.min() + yrs_ago <= x["Year"].max() <= df.Year.max()
    )
    filtered_group[new_col_name] = avg_attendance.shift(yrs_ago)
    return filtered_group


final_feature_dfs = []
for y in range(1, YEARS_AGO + 1):
    df_with_avg_attendance = get_avg_attendance(df=df, yrs_ago=y).copy()
    df_with_avg_attendance = df_with_avg_attendance
    
    feature_df = []
    for feature in FEATURES:
        df_with_final_feature = get_previous_year_metrics(
        df=df, feature=feature, yrs_ago=y).copy()
        feature_df.append(df_with_final_feature[["Date", "HomeTeam", "VisitingTeam", f"final_{feature}_{y}_yr_ago"]])
    feature_df = pd.concat(feature_df, axis=1)
    
    d = pd.concat([df_with_avg_attendance, feature_df], axis=1)
    final_feature_dfs.append(d.loc[:, ~d.columns.duplicated()])
    
final_feature_dfs = pd.concat(final_feature_dfs, axis=1)
# filter by threshold of not null values
final_feature_dfs.dropna(
    axis=1, thresh=int(final_feature_dfs.shape[0] * 0.8), inplace=True
)

final_feature_dfs.dropna(axis=0, inplace=True)

final_feature_dfs = final_feature_dfs.loc[:, ~final_feature_dfs.columns.duplicated()]

df = final_feature_dfs.copy()

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

# gmaps = googlemaps.Client(key=API_KEY)

# Geocoding an address
# city_coord = {"lat": [], "lng": []}
# for city in df["HomeTeam_City"].unique():
#     geocode_result = gmaps.geocode(city)
#     res = geocode_result[0]["geometry"]["location"]
#     city_coord["lat"].append(res["lat"])
#     city_coord["lng"].append(res["lng"])

# city_coord = pd.DataFrame(city_coord, index=np.arange(len(city_coord["lat"])))

# city_coord = pd.concat([pd.Series(df["HomeTeam_City"].unique()), city_coord], axis=1)

# city_coord.to_csv(DATA_PATH.joinpath("raw", "city_coords.csv"))


def get_weather(df):
    weather_df = pd.DataFrame()
    print("Gathering weather metrics.")
    for (lt, lg), g in df.groupby(["lat", "lng"]):
        g = g.sort_values(by=["Date"])
        start_date = g["Date"].iat[0].strftime("%Y-%m-%d")
        end_date = g["Date"].iat[-1].strftime("%Y-%m-%d")
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lt}&longitude={lg}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant&timezone=America%2FNew_York&temperature_unit=fahrenheit&windspeed_unit=mph&precipitation_unit=inch"
        payload = {}
        headers = {}
        response = requests.request("GET", url, headers=headers, data=payload)
        res = response.json()
        w = pd.DataFrame(res["daily"])
        w["Date"] = pd.to_datetime(w["time"], format="%Y-%m-%d")
        w = pd.merge(
            left=w,
            right=g[["Date", "HomeTeam", "VisitingTeam"]],
            how="inner",
            on="Date",
        )
        weather_df = pd.concat([weather_df, w], axis=0)
    return weather_df


weather = get_weather(df)

df = pd.merge(
    left=df, right=weather, how="inner", on=["Date", "HomeTeam", "VisitingTeam"]
).reset_index(drop=True)

# load income and population data
demographics = pd.read_excel(DATA_PATH.joinpath("processed", "City_data.xlsx"))

demographics = demographics.rename(
    {"year": "Year", "city_name": "HomeTeam_City"}, axis=1
)

df = df.merge(
    demographics[["Year", "HomeTeam_City", "avg_income", "population"]],
    on=["Year", "HomeTeam_City"],
    how="inner",
)

# drop any duplicate rows
df = df.loc[~df.duplicated(), :]

# print(df.shape)
# print(df.head())
# print(df.columns.tolist())


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
    "HomeTeam_Rank",
    # "HomeTeam_W",
    # "temperature_2m_max",
    # "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "windspeed_10m_max",
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

attendance_features = [f"avg_attendance_{y}_yr_ago" for y in range(1, YEARS_AGO + 1)]

historical_rankings = []
for y in range(1, YEARS_AGO + 1):
    for feature in FEATURES:
        historical_rankings.append(f"final_{feature}_{y}_yr_ago")

CONT_FEATURES.extend(attendance_features)
CONT_FEATURES.extend(historical_rankings)


CAT_FEATURES = [
    "BallParkID",
    "HomeTeam_City",
    "Dayofweek",
    "VisitingTeam",
    "VisitingTeamLeague",
    "HomeTeam",
    "HomeTeamLeague",
]

splits = RandomSplitter(valid_pct=0.2)(range_of(df))

to = TabularPandas(
    df,
    procs=[Categorify, FillMissing, Normalize],
    cat_names=CAT_FEATURES,
    cont_names=CONT_FEATURES,
    y_names="Attendance_TRUTH_y",
    splits=splits,
)

dls = to.dataloaders(bs=64)

cardinalities = df[CAT_FEATURES].nunique().to_numpy()
emb_szs = {cat: min(50, card//2) for cat, card in zip(CAT_FEATURES, cardinalities)}

config = tabular_config(y_range=[0, 75000], act_cls=nn.GELU())
learn = tabular_learner(dls, metrics=mae, layers=[1000, 500, 250], emb_szs=emb_szs, config=config)  # default is [200, 100]
learn.fit_one_cycle(25)


# function to embed features ,obtained from fastai forums
def embed_features(learner, xs):
    xs = xs.copy()
    for i, feature in enumerate(learner.dls.cat_names):
        emb = learner.model.embeds[i]
        new_feat = pd.DataFrame(
            emb(tensor(xs[feature], dtype=torch.int64, device="mps")),
            index=xs.index,
            columns=[f"{feature}_{j}" for j in range(emb.embedding_dim)],
        )
        xs.drop(columns=feature, inplace=True)
        xs = xs.join(new_feat)
    return xs


embeddings = embed_features(learn, to.all_cols)

embeddings.to_parquet(DATA_PATH.joinpath("processed", "train_embed.parquet"))
