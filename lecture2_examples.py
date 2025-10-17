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
cnt_diff = (sinx[2:] - sinx[:-2]) / (2*dx) # CUNT diff

plt.plot(x, cosx, label=r'Analaytical Derivative of $\sin{x}$')
plt.plot(x[:-1], fwd_diff, label='Forward Diff Approx')
plt.plot(x[1:], bkd_diff, label='Backward Diff Approx')
plt.plot(x[1:-1], cnt_diff, label='Central Diff Approx')
plt.legend(loc='best')

dxs = [2**-n for n in range(20)] # list comprehensionn builds list real fast
# start doing this luile
# wow cool coding trick for compressing thing
err_fwd, err_cnt = [], []

for dx in dxs:
    x = np.arange(0, 2.5*np.pi, dx)

    sinx = np.sin(x)

    #doin this the easy way because i love my computer
    fwd_diff = (sinx[1:] - sinx[:-1]) / dx
    cnt_diff = (sinx[2:] - sinx[:-2]) / (2*dx) # CUNT diff  

    err_fwd.append(np.abs(fwd_diff[-1] - np.cos(x[-1])))
    err_cnt.append(np.abs(cnt_diff[-1] - np.cos(x[-2])))


figgy, axy = plt.subplots(1,1)
axy.loglog(dxs, err_fwd, '-.', label='Forward Diff', lw=5)
# slope of line matches power of the error (2ln(dx))
# lower end errors problem with machine precision: cant get
# meaning lower numbers
axy.loglog(dxs, err_cnt, '-.', label='Central Diff', lw=5)
axy.legend(loc='best')
axy.set_xlabel(r'$\Delta x$')
axy.set_ylabel('Error')