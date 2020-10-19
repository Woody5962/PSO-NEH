import numpy as np
from PSO import pso
from M_NEH import m_neh
import time

PT = np.array([[45, 48, 50, 35, 35, 30, 30, 35, 25, 26],
               [45, 50, 45, 35, 36, 35, 35, 34, 25, 30],
               [50, 45, 46, 35, 36, 36, 31, 34, 30, 31],
               [50, 48, 48, 34, 38, 35, 32, 33, 27, 31],
               [45, 46, 48, 30, 35, 50, 34, 32, 28, 31],
               [45, 45, 45, 30, 35, 50, 33, 32, 30, 26],
               [47, 50, 47, 31, 30, 35, 35, 31, 29, 25],
               [50, 45, 48, 32, 30, 34, 34, 30, 24, 27],
               [48, 46, 46, 33, 34, 30, 34, 30, 25, 25],
               [45, 47, 47, 33, 33, 30, 35, 34, 32, 26],
               [46, 50, 45, 34, 30, 50, 30, 35, 31, 25],
               [48, 50, 47, 35, 31, 35, 32, 30, 25, 30]])
mop = [3, 3, 2, 2]

start = time.time()
g, fg = pso(m_neh, PT, mop, debug=True)
end = time.time()
print()

print('The optimum is at:')
print('{}'.format(g))
while m_neh(g, PT, mop, False, fg) != fg:
    pass
print('The processing time is : \n', fg)
