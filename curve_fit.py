import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def cos_func(times, amplitude_1, frequency_1):
    # return amplitude_1 * np.cos(frequency_1 * times)
    # return amplitude_1 * np.sin(frequency_1 * times)
    return amplitude_1 * np.sin(frequency_1 * times) + amplitude_2 * np.cos(frequency_2 * times)

def decaying_sinusoid(t, a, lam, w):
    # return a * np.exp(lam * t)
    return a * np.exp(lam * t) * np.cos(w * t)

y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/here_1_x.csv")
xdata = np.array(list(range(len(y))))

popt, pcov = curve_fit(decaying_sinusoid,  # our function
                       xdata,  # measured x values
                       y,  # measured y values
                       )
# p0=(3.0, period2freq(0.44)))  # the initial guess for the two parameters


print(popt)

fig, ax = plt.subplots(1, 1)
ax.plot(xdata, y, '.', label='Measured')
ax.plot(xdata[:80], decaying_sinusoid(xdata[:80], popt[0], popt[1], popt[2]), label='Best Fit')
ax.legend()
plt.show()
