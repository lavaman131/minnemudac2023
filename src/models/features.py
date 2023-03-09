import numpy as np

FEATURES = np.array(['HomeTeamOffense_IntentionalWalks', 'HomeTeamOffense_Strickouts',
                    'Rank', 'W', 'VisitingTeamPitchers_TeamEarnedRuns',
                    'HomeTeamOffense_SacrificeHits', 'Gm#', 'LengthofGame',
                    'VisitingTeamPitchers_IndividualEarnedRuns', 'cLI',
                    'HomeTeamOffense_Hits', 'HomeTeamScore', 'BallParkID',
                    'VisitingTeamOffense_SacrificeHits', 'R', 'DayofWeek', 'DayNight',
                    'Streak_count', 'HomeTeamOffense_RBIs', 'Attendance'], dtype='<U41')

# decision tree params
DT_PARAMS = {'max_depth': 5, 'max_leaf_nodes': 19}

# random forest params
RF_PARAMS = {'max_depth': 5, 'max_leaf_nodes': 19, 'n_estimators': 100}

# xgboost params
XGB_PARAMS = {'max_depth': 5, 'max_leaves': 19, 'learning_rate': 0.5, 'n_estimators': 100}
