import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

xdata_1 = np.array([420, 367, 348, 332, 315, 296, 295]) * 10**-2
ydata_1 = np.array([8215, 11048, 12321, 14334, 16714, 20161, 20647]) *10**-2

print(xdata_1)
print(ydata_1)