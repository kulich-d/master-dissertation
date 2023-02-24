import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
d1 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_heel.txt")[:, 0]
d2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")[:, 0]

y1 = savgol_filter(d1, 15, 2, mode='interp')
y2 = savgol_filter(d2, 15, 2, mode='interp')

df_interpolated = pd.DataFrame({'S1_Joy': d1, 'S2_Joy': d2})

# Set window size to compute moving window synchrony.
r_window_size = 20
# Interpolate missing data.
# df_interpolated = df.interpolate()
# Compute rolling window synchrony
rolling_r = df_interpolated['S1_Joy'].rolling(window=r_window_size, center=True).corr(df_interpolated['S2_Joy'])
f, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
rolling_r = rolling_r.dropna()
df_interpolated.rolling(window=5, center=True).median().plot(ax=ax[0])
ax[0].set(xlabel='Frame', ylabel='Smiling Evidence')
rolling_r.plot(ax=ax[1])
ax[1].set(xlabel='Frame', ylabel='Pearson r')
plt.suptitle("Smiling data and rolling window correlation")
plt.show()
