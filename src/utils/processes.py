import pandas as pd
import numpy as np
from typing import List

def to_numerical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    data = np.empty((df.shape[0], len(columns)))
    for idx, col in enumerate(df[columns]):
        mapping = {key: idx for idx, key in enumerate(np.unique(df[col]))}
        data[:, idx] = df[col].map(mapping)
    return pd.DataFrame(data, columns=columns)
        
    