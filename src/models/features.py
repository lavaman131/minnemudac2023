import numpy as np



# looking if teams are rivals
# WAR
# runs per game 
# average age of team (pitch age, bat_age)
# all-stars

FEATURES = np.array(
    [
        "avg_attendance_1_yr_ago",
        "avg_attendance_2_yr_ago",
        "avg_attendance_3_yr_ago",
        "is_holiday",
        "Year",
        "Month",
        "Week",
        "Dayofweek",
        "DayNight",
        "Dayofyear",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
        "Stadium_Capacity",
        "BallParkID",
        "HomeTeam_cLI",
        "HomeTeam_Rank",
        "HomeTeam_W",
        "HomeTeam_Streak_count",
        "HomeTeamGameNumber",
        "VisitingTeam_cLI",
        "VisitingTeam_Rank",
        "VisitingTeam_L",
        "VisitingTeam_Streak_count",
        "VisitingTeamGameNumber",
    ]
)

# decision tree params
DT_PARAMS = {"max_depth": 11, "max_leaf_nodes": 75}

# random forest params
RF_PARAMS = {"max_depth": 11, "max_leaf_nodes": 75, "n_estimators": 95}

# xgboost params
XGB_PARAMS = {
    "max_depth": 11,
    "max_leaves": 75,
    "learning_rate": 0.14,
    "reg_lambda": 3,
    "n_estimators": 1000,
}

# catboost params
CB_PARAMS = {
    "grow_policy": "Lossguide",
    "depth": 11,
    "max_leaves": 75,
    "n_estimators": 1000,
    "learning_rate": 0.14,
    "l2_leaf_reg": 3,
}
