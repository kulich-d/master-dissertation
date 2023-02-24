import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator, BSpline

y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")[:, 1]
x = np.array(list(range(len(y))))

spl = CubicSpline(x, y, bc_type = 'natural')

plt.plot(x, y, "o", label="observation")
plt.plot(x, spl(x), label="CubicSpline")
# plt.plot(x, Akima1DInterpolator(x, y)(x), '-', label='Akima1D')
# plt.plot(x, PchipInterpolator(x, y)(x), '-', label='Akima1D')
plt.plot(x, BSpline(x, y, 2)(x), '-', label='BSpline')

plt.legend()
plt.show()
