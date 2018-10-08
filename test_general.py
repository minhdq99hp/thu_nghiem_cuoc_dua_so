diff_up = a - up
diff_down = a - down

diff_up[diff_up >= 0] = 1
diff_up[diff_up < 0] = 0

diff_down[diff_down <= 0] = -2
diff_down[diff_down > 0] = 0
diff_down[diff_down == -2] = 1


print((diff_up + diff_down) * a)

