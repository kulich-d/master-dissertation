from scipy import fftpack

import matplotlib.pyplot as plt
import numpy as np

y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/here_1_x.csv")
x = np.array(list(range(len(y))))

ft_populations = fftpack.fft(y, axis=0)
frequencies = fftpack.fftfreq(y.shape[0], x[1] - x[0])
periods = 1 / frequencies

plt.figure()
plt.plot(periods, abs(ft_populations) * 1e-3, 'o')
plt.xlim(0, 22)
plt.xlabel('Period')
plt.ylabel('Power ($\cdot10^3$)')

plt.show()