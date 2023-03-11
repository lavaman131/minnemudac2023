from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import csv
from pathlib import Path
from tqdm import tqdm
from typing import List
import numpy as np


OPTIONS = Options()
OPTIONS.add_argument('--headless')

# helper functions
has_index = lambda x: x[0] != ','
        
        
def get_team_standings(team_names: List[str], seasons: List[int], save_path: str='../../data/raw/standings', max_retries=3):
    seasons = sorted(seasons)
    
    for team_name in tqdm(team_names, colour='#006D5B'):
        print(f'Collecting standings for {team_name}.')
        standings = []
        for season in tqdm(seasons, colour='#ffffff'):
            driver = webdriver.Firefox(options=OPTIONS)
            driver.install_addon('/Users/alilavaee/Library/Application Support/Firefox/Profiles/shh78awq.default-release/extensions/uBlock0@raymondhill.net.xpi', temporary=True)
            url_team_name = team_name
            # LAA was called ANA before 2005
            if team_name == 'ANA' and season > 2004:
                url_team_name = 'LAA'
            # MIA was called FLO (FLA) before 2012 
            if team_name == 'MIA' and season <= 2011:
                url_team_name = 'FLA'
            # TBA was called TBD before 2008
            if team_name == 'TBA' and season <= 2007:
                url_team_name = 'TBD'
            # WAS/WSN was called MON before 2005
            if team_name == 'WAS' and season <= 2004:
                url_team_name = 'MON'
            elif team_name == 'WAS' and season > 2004:
                url_team_name = 'WSN'
            if team_name == 'KCA':
                url_team_name = 'KCR'
            if team_name == 'SLN':
                url_team_name = 'STL'
                
            tries = 0
            while True:
                tries += 1
                try:
                    driver.get(f'https://www.baseball-reference.com/teams/{url_team_name}/{season}-schedule-scores.shtml#all_team_schedule')
                    driver.install_addon('/Users/alilavaee/Library/Application Support/Firefox/Profiles/shh78awq.default-release/extensions/uBlock0@raymondhill.net.xpi', temporary=True)
                    e1 = driver.find_element(By.CSS_SELECTOR, '#team_schedule_sh > div:nth-child(3) > ul:nth-child(1) > li:nth-child(1) > span:nth-child(1)')
                    action1 = ActionChains(driver).move_to_element(e1).click(e1)
                    action1.perform()
                    e2 = driver.find_element(By.CSS_SELECTOR, '#team_schedule_sh > div:nth-child(3) > ul:nth-child(1) > li:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(3) > button:nth-child(1)')
                    action2 = ActionChains(driver).move_to_element(e2).click(e2)
                    action2.perform()
                except NoSuchElementException:
                    driver.quit()
                    driver = webdriver.Firefox(options=OPTIONS)
                    if tries == max_retries:
                        raise ValueError('Maximum retries exceeded.')
                else:
                    break
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            driver.quit()
            res = soup.find('pre', attrs={'id': 'csv_team_schedule'}).text.split('\n')
            # start at header of csv
            res = res[4:]
            # remove invalid rows (without index)
            res = [r for r in res if has_index(r)]
            res = [r.split(',') for r in res] 
            # add year to header
            res[0].insert(1, 'Year')
            for i in range(1, len(res)):
                res[i].insert(1, season)
            standings.extend(res)
            
        with open(Path(save_path)/f'{team_name}_{seasons[0]}-{seasons[-1]}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', 
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(standings)
            
if __name__ == '__main__':
    mlb_team_ids = ['ANA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHA', 'CHN', 'CIN', 'CLE', 'COL', 
                'DET', 'HOU', 'KCA','LAN', 'MIA', 'MIL', 'MIN', 'NYA',
                'NYN', 'OAK', 'PHI', 'PIT', 'SDN', 'SEA', 'SFN', 'SLN', 'TBA', 'TEX',
                'TOR', 'WAS']
    get_team_standings(team_names=mlb_team_ids, seasons=np.arange(2000, 2023))