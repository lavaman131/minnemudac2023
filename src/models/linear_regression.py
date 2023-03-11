import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from src.models.features import FEATURES


DATA_PATH = Path('../../data')
df = pd.read_parquet(DATA_PATH.joinpath('processed', 'train_v2.parquet'))

X = df.drop(['Attendance_TRUTH_y'], axis=1).to_numpy()
y = df['Attendance_TRUTH_y'].to_numpy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression().fit(X_train, y_train)
r_2 = reg.score(X_train, y_train)
print(f'R squared: {r_2}')
y_pred = reg.predict(X_val)
MAE = np.abs(y_pred - y_val).mean()
print(f'Validation MAE: {MAE} compared to pure randomness: {np.abs(y_val).mean()}')