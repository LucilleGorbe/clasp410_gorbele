#!/usr/bin/env python3


'''
Series of simple examples for Lecture 2 about turkeys
'''

import numpy as np
import matplotlib.pyplot as plt

dx = 0.1
x = np.arange(0, 6*np.pi, dx)
sinx = np.sin(x)
cosx = np.cos(x) # analytical solution

#doin this the easy way because i love my computer
fwd_diff = (sinx[1:] - sinx[:-1]) / dx
bkd_diff = (sinx[1:] - sinx[:-1]) / dx

plt.plot(x, cosx, label=r'Analaytical Derivative of $\sin{x}$')
plt.plot(x[:-1], fwd_diff, label='Forward Diff Approx')
plt.plot(x[1:], bkd_diff, label='Backward Diff Approx')
plt.legend(loc='best')

