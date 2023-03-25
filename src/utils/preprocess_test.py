import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset

# create a sample DataFrame with a "Year" column
df = pd.DataFrame(
    {
        "Year": [2018, 2018, 2020, 2020, 2019, 2019],
        "Value": [10, 20, 30, 40, 50, 60],
        "HomeTeam": ["D", "D", "B", "B", "D", "D"],
        "Dummy": ["D", "D", "B", "B", "D", "D"]
    }
)

group = df.groupby(["Year", "HomeTeam"])
for idx, row in df.iterrows():
    try:
        prev_metric = group.get_group((row.Year - 1, row.HomeTeam))["Value"].mean()
        df.loc[idx, "Value_Prev"] = prev_metric
    except: continue
    
print(df)
    
