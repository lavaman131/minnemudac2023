import pandas as pd
import numpy as np
from pathlib import Path
from src.data.metadata import get_team_standings

mlb_team_ids = ['ANA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHA', 'CHN', 'CIN', 'CLE', 'COL', 
                'DET', 'HOU', 'KCA','LAN', 'MIA', 'MIL', 'MIN', 'NYA',
                'NYN', 'OAK', 'PHI', 'PIT', 'SDN', 'SEA', 'SFN', 'SLN', 'TBA', 'TEX',
                'TOR', 'WAS']
get_team_standings(team_names=mlb_team_ids, seasons=np.arange(2000, 2023))