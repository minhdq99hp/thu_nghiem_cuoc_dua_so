import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def func(x, a, b):
    return (x + a)/(x + b)


def fit_curve_based_on_ground_data(file_name):
    ground_data = open('ground_data.txt', 'r')

    xdata_1 = np.array([int(x) for x in ground_data.readline().split(' ')])
    ydata_1 = np.array([int(y) for y in ground_data.readline().split(' ')])

    # p0 = 1718000, -210
    p0 = 786, -14
    popt, pcov = curve_fit(func, xdata_1, ydata_1, p0)

    return popt, pcov

    # print(popt)
    # [ 1.94926806e+06 -1.90139644e+02]

    # xdata = [x for x in range(211, 419)]
    # plt.plot(xdata, func(xdata, *popt), 'g--', label='fitted_curve')
    # plt.plot(xdata_1, ydata_1, 'r-', label='origin_data')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()
