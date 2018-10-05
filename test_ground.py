import matplotlib.pyplot as plt

x = [420, 367, 348, 332, 315, 296, 295]
y = [8215, 11048, 12321, 14334, 16714, 20161, 20647]

all_x = [x for x in range(295, 421, 5)]
all_y = [(x+1718000)/(x-210) for x in range(295, 421, 5)]

plt.plot(x, y)
plt.plot(all_x, all_y, 'o')

plt.show()