from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return np.corrcoef(datay.shift(lag))


d1 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_heel.txt")[:, 0]
d2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")[:, 0]

y1 = savgol_filter(d1, 15, 2, mode='interp')
y2 = savgol_filter(d2, 15, 2, mode='interp')

seconds = 5
fps = 30
rs = [crosscorr(d1, d2, lag) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]
offset = np.floor(len(rs) / 2) - np.argmax(rs)
f, ax = plt.subplots(figsize=(14, 3))
ax.plot(rs)
ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', ylim=[.1, .31], xlim=[0, 301], xlabel='Offset',
       ylabel='Pearson r')
ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
plt.legend()