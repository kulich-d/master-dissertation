import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")[:, 0][:100]
x = np.array(list(range(len(y))))

spl = UnivariateSpline(x, y)

from scipy.signal import savgol_filter

#
# plt.plot(x, y, "o", label="observation")
# y_filter = savgol_filter(y, 15, 2, mode='interp')
# plt.plot(x, y_filter, 'g', lw=3)
# print(y_filter)


# # plt.plot(x, spl(x), label="CubicSpline")
# # plt.plot(x, Akima1DInterpolator(x, y)(x), '-', label='Akima1D')
# # plt.plot(x, PchipInterpolator(x, y)(x), '-', label='Akima1D')
# plt.plot(x, BSpline(x, y, 2)(x), '-', label='BSpline')
#
# plt.legend()
# plt.show()

# #  it's good
from scipy.fft import dct, idct

# число точек в normalized_tone
dct = dct(y, norm="ortho")
dct[9:] = 0
smoothed = idct(dct, norm="ortho")
#
plt.scatter(x, y, )
plt.plot(x, smoothed, label = "FFT")
plt.legend()

plt.show()
# #

