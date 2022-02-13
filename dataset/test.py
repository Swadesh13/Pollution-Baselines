#%%
import pandas as pd
import numpy as np
import os

params = ['PM2.5', 'PM10', 'CO', 'NO2', 'Ozone', 'WD', 'WS', 'SR', 'RH']
cols = []
for param in params:
    c = pd.read_csv(f'{param}.csv').columns
    cols.append(c)
    print(len(c))

d = cols[0]
for c in cols:
	d = list(set(d) & set(c))
print(f"Selecting data for the following {len(d)} stations:", d)
