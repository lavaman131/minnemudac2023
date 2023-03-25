import numpy as np

YEARS_AGO = 2

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
]

LAST_FEATURES = [
    "HomeTeam_cLI",
    "HomeTeam_Rank",
    "HomeTeam_W",
    "VisitingTeam_cLI",
    "VisitingTeam_Rank",
    "VisitingTeam_W",
]

AVG_FEATURES = ["Attendance_TRUTH_y"]

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


# decision tree params
DT_PARAMS = {"max_depth": 12, "max_leaf_nodes": 75}

# random forest params
RF_PARAMS = {"max_depth": 12, "max_leaf_nodes": 75, "n_estimators": 95}

# xgboost params
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.01,
}

# catboost params
CB_PARAMS = {
    "n_estimators": 1400,
    # "learning_rate": 0.225,
}
