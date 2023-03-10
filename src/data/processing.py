import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

def clean_files(p: Union[str, Path]='../../data/raw/standings'):
    # convert to Path object if string
    if not isinstance(p, Path):
        p = Path(p)
    for f in p.iterdir():
        if f.suffix == '.csv':
            with open(f, 'r') as file:
                lines = file.readlines()
            with open(f, 'w') as file:
                for idx, line in enumerate(lines):
                        if idx == 0 or line[0] != 'G':
                            file.write(line)
                
def _get_home_away_col(df):
    away = []
    home = []
    mapping = {'SDP': 'SDN', 'STL': 'SLN', 'CHW': 'CHA', 'MIL': 'MIL', 'BAL': 'BAL',
               'TEX': 'TEX', 'TOR': 'TOR', 'KCR': 'KCA', 'HOU': 'HOU', 'ATL': 'ATL',
               'PIT': 'PIT', 'MON': 'MON', 'WSN': 'WAS', 'SFG': 'SFN', 'FLA': 'FLO',
               'MIA': 'MIA', 'NYY': 'NYA', 'DET': 'DET', 'COL': 'COL', 'CLE': 'CLE',
               'BOS': 'BOS', 'CIN': 'CIN', 'LAD': 'LAN', 'ARI': 'ARI', 'TBD': 'TBA',
               'TBR': 'TBA', 'SEA': 'SEA', 'CHC': 'CHN', 'ANA': 'ANA', 'LAA': 'ANA',
               'PHI': 'PHI', 'OAK': 'OAK', 'NYM': 'NYN', 'MIN': 'MIN'}
    df['Tm'] = df['Tm'].map(mapping)
    df['Opp'] = df['Opp'].map(mapping)
    for _, row in df.iterrows():
        away.append(row['Tm'] if row['At'] == '@' else row['Opp'])
        home.append(row['Opp'] if row['At'] == '@' else row['Tm'])
    df['VisitingTeam'] = away
    df['HomeTeam'] = home
      
def clean_csv(p: Union[str, Path]='../../data/raw/standings', 
              year_range: str='2000-2022',
              save_file_path: Union[str, Path]='../../data/processed'):
    if not isinstance(p, Path):
        p = Path(p)
    if not isinstance(save_file_path, Path):
        save_file_path = Path(save_file_path)
    full_df = pd.DataFrame(columns=['Gm#', 'Year', 'Date', 'Unnamed: 3', 
                                    'Tm', 'Unnamed: 5', 'Opp', 'W/L',
                                    'R', 'RA', 'Inn', 'W-L', 'Rank', 'GB', 'Win',
                                    'Loss', 'Save', 'Time', 'D/N', 'Attendance_TRUTH',
                                    'cLI', 'Streak', 'Orig. Scheduled'])
    for f in p.iterdir():
        if f.suffix == '.csv':
            df = pd.read_csv(f)
            df.drop_duplicates(keep='first', inplace=True)
            day_month = df.Date.apply(lambda s: ' '.join(s.split(' ')[1:3]))
            full_date = day_month + ' ' + df.Year.astype('str')
            full_date = pd.to_datetime(full_date, format='%b %d %Y')
            df.insert(loc=1, column='Full_Date', value=full_date)
            df.drop(['Year', 'Date'], axis=1, inplace=True)
            df.rename({'Full_Date': 'Date', 'Unnamed: 5': 'At', 
                       'Attendance': 'Attendance_TRUTH'}, axis=1, inplace=True)
            _get_home_away_col(df)
            full_df = pd.concat([full_df, df], axis=0)   
             
    # Where Tm == HomeTeam
    df_home = full_df.loc[full_df.Tm == full_df.HomeTeam].copy()
    # collect wins and losses for home team
    df_home['HomeTeam_W'] = df_home.loc[:, 'W-L'].apply(lambda s: s.split('-')[0]).astype('int')
    df_home['HomeTeam_L'] = df_home.loc[:, 'W-L'].apply(lambda s: s.split('-')[1]).astype('int')
    df_home.rename({'Rank': 'HomeTeam_Rank'}, axis=1, inplace=True)
    df_home.rename({'cLI': 'HomeTeam_cLI'}, axis=1, inplace=True)
    # convert streak to quantitative
    df_home.loc[:, 'HomeTeam_Streak_count'] = df_home['Streak'].apply(lambda s: str(s).count('+') 
                                                                if str(s).startswith('+')
                                                                else -1 * str(s).count('-'))
    
    # Where Tm == VisitingTeam
    df_away = full_df.loc[full_df.Tm == full_df.VisitingTeam].copy()
    # collect wins and losses for away team
    df_away['VisitingTeam_W'] = df_away.loc[:, 'W-L'].apply(lambda s: s.split('-')[0]).astype('int')
    df_away['VisitingTeam_L'] = df_away.loc[:, 'W-L'].apply(lambda s: s.split('-')[1]).astype('int')
    df_away.rename({'Rank': 'VisitingTeam_Rank'}, axis=1, inplace=True)
    df_away.rename({'cLI': 'VisitingTeam_cLI'}, axis=1, inplace=True)
    # convert streak to quantitative
    df_away['VisitingTeam_Streak_count'] = df_away.loc[:, 'Streak'].apply(lambda s: str(s).count('+')
                                                                if str(s).startswith('+')
                                                                else -1 * str(s).count('-'))
    cols = ['Date', 'HomeTeam', 'VisitingTeam', 'VisitingTeam_W', 'VisitingTeam_L', 
             'VisitingTeam_Rank', 'VisitingTeam_cLI',
             'VisitingTeam_Streak_count', 'Attendance_TRUTH']
    full_df = pd.merge(df_home, df_away[cols], how='inner', on=['Date', 'HomeTeam', 'VisitingTeam'])
    print(full_df.shape)
    full_df.to_parquet(save_file_path/f'All_Team_Standings_{year_range}.parquet')
    print('Done')
# clean_files()
clean_csv()


# mapping scraped data to game logs
# ['SDP'] -> ['SDN']
# ['STL'] -> ['SLN']
# ['CHW'] -> ['CHA']
# ['MIL'] -> ['MIL']
# ['BAL'] -> ['BAL']
# ['TEX'] -> ['TEX']
# ['TOR'] -> ['TOR']
# ['KCR'] -> ['KCA']
# ['HOU'] -> ['HOU']
# ['ATL'] -> ['ATL']
# ['PIT'] -> ['PIT']
# ['MON' 'WSN'] -> ['MON', 'WAS']
# ['SFG'] -> ['SFN']
# ['FLA' 'MIA'] -> ['FLO', 'MIA']
# ['NYY'] -> ['NYA']
# ['DET'] -> ['DET']
# ['COL'] -> ['COL']
# ['CLE'] -> ['CLE']
# ['BOS'] -> ['BOS']
# ['CIN'] -> ['CIN']
# ['LAD'] -> ['LAN']
# ['ARI'] -> ['ARI']
# ['TBD' 'TBR'] -> ['TBA']
# ['SEA'] -> ['SEA']
# ['CHC'] -> ['CHN']
# ['ANA' 'LAA'] -> ['ANA']
# ['PHI'] -> ['PHI']
# ['OAK'] -> ['OAK']
# ['NYM'] -> ['NYN']
# ['MIN'] -> ['MIN']





        

    