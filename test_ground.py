import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


# xdata = np.linspace(0, 4, 50)
# y = func(xdata, 2.5, 1.3, 0.5)
# np.random.seed(1729)
# y_noise = 0.2 * np.random.normal(size=xdata.size)
# ydata = y + y_noise

xdata_1 = np.array([420, 367, 348, 332, 315, 296, 295])
ydata_1 = np.array([8215, 11048, 12321, 14334, 16714, 20161, 20647])

# xdata = np.linspace(295, 420)
# ydata = func2(xdata, 1718000, - 210)

# plt.plot(xdata, ydata, 'b-', label='data')

plt.plot(xdata_1, ydata_1, 'r-', label='origin_data')
# popt, pcov = curve_fit(func, xdata_1, ydata_1, diag=(1./xdata_1.mean(),1./ydata_1.mean()))
#
# plt.plot(xdata_1 * 10 , func(xdata_1, *popt) * 10, 'b-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
# plt.plot(xdata, func(xdata, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


all_x = [x for x in range(295, 421, 5)]
all_y = [(x+1718000)/(x-210) for x in range(295, 421, 5)]

plt.plot(all_x, all_y, 'go')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
