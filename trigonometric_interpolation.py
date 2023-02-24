import numpy as np
import torch
from scipy.fftpack import fft


def _get_inter_coeffs(x: torch.Tensor) -> np.array:
    y = fft(x.cpu().numpy())
    a, b = y.real, y.imag
    return a, b


def trig_interpolate(xn: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    x0 = x[0].item()

    L = (x[-1] - x0).item()

    a, b = _get_inter_coeffs(u)

    N = len(a)
    xn = xn.cpu().numpy()

    d = xn - x0

    un = np.zeros_like(xn)

    for k, (ak, bk) in enumerate(zip(a, b)):
        un += ak * np.cos(2 * np.pi * k * (d / L)) - bk * np.sin(2 * np.pi * k * (d / L))
    return (1 / N) * torch.Tensor(un)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/here_1_x.csv")
    x = np.array(list(range(len(y))))
    fs = 8
    Y = np.fft.fft(y)
    freq = np.fft.fftfreq(x.shape[-1], 1 / fs)

    upSampRatio = 5

    fsNew = fs * upSampRatio
    xInterps = np.arange(2 * fsNew) / fsNew

    Y = np.concatenate([Y, [np.conjugate(Y[fs])]])
    freq = np.concatenate([freq, [-freq[fs]]])
    print(freq)

    yInterp = np.array([
        np.dot(Y, np.exp(freq * 2j * np.pi * xInterp) / x.shape[-1])
        for xInterp in xInterps
    ])

    plt.figure(figsize=[15, 5])
    plt.plot(x, y)
    plt.plot(xInterps, yInterp.real)
    plt.xlabel('Time [sec]')
    plt.legend(['Original', 'Resampled'])
    plt.grid()
    plt.show()
