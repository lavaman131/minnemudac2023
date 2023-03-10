import numpy as np

# categorical embeddings (for day of the week, player names/IDs, Ball park ID),
# incorporating average temperature/weather into the model,
# looking at if a day of a game is an observed holiday
FEATURES = [
    "avg_attendance_1_yr_ago",
    "StadiumID",
    "Stadium_Capacity",
    "Year",
    "Month",
    "is_holiday",
    "DayNight",
    "HomeTeam_cLI",
    "HomeTeam_Rank",
    "HomeTeam_W",
    "HomeTeam_Streak_count",
    "HomeTeamGameNumber",
    "VisitingTeam_Rank",
    "VisitingTeam_L",
    "VisitingTeam_Streak_count",
    "VisitingTeamGameNumber",
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
]

# decision tree params
DT_PARAMS = {"max_depth": 11, "max_leaf_nodes": 75}

# random forest params
RF_PARAMS = {"max_depth": 11, "max_leaf_nodes": 75, "n_estimators": 95}

# xgboost params
XGB_PARAMS = {
    "max_depth": 10,
    "max_leaves": 75,
    "learning_rate": 0.14,
    "reg_lambda": 3,
    'n_estimators': 1000
}

# catboost params
CATB_PARAMS = {
    "depth": 10,
    "n_estimators": 1000,
    "learning_rate": 0.14,
    "l2_leaf_reg": 3,
}
