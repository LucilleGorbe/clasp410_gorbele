#!/usr/bin/env pyton3
'''
Solve the coffee problem to learn how to drink coffee effectively.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def solve_temp(t, Tenv=20., T0=90., k=1/300.):
    '''
    This function returns temperature as a function of time using Newton's Law of Cooling.
    ---------
    Parameters
    ---------
    Tenv - float
        ambient temperature of the air environment
    T0 - float
        starting temperature of the coffee
    k - float
        heat transfer coefficient of the coffee
    t - vector of float
        timeline for which the cooling occurs
    ---------
    Returns
    ---------
    T_coffee - vector of floats
        
    '''
    T_coffee = Tenv + (T0-Tenv)*np.exp(-k*t)

    return T_coffee

def time_to_temp(Tfinal, Tenv=20., T0=90., k=1/300.):
    '''
    This function returns the exact time that the coffee reaches a final, 
    drinkable temperature using Newton's Law of Cooling for maximum sippage.
    ---------
    Parameters
    ---------
    Tfinal - float
        target temperature of coffee
    Tenv - float
        ambient temperature of the air environment
    T0 - float
        starting temperature of the coffee
    k - float
        heat transfer coefficient of the coffee
    ---------
    Returns
    ---------
    t - float
        time in seconds to cool to Tfinal
    '''
    #input the equation all good style
    t = -(1/k)*np.log((Tfinal - Tenv)/(T0 - Tenv))

    return t

def verify_code():
    '''
    Verify that the code is implemented correct
    '''

    t_real = 60. * 10.76
    k = np.log(95./110.) / -120.
    t_code = time_to_temp(120., T0=180., Tenv = 70., k=k)

    print(f"Target solution is: {t_real}. \nNumerical solution is: {t_code}. \
          \nDifference is: {t_real-t_code}")

#test different scenarios
#do it quantitatively to sceen first:
t1 = time_to_temp(65)           # add cream at T=65 to get to 60
t2 = time_to_temp(60, T0=85)    # add cream right away
t3 = time_to_temp(60)           # control case, no cream

print(f"Times to drinkable coffee:\n\t \
        Cream added at T=65: {t1:.2f}s\n\t \
        Cream added immediately: {t2:.2f}s\n\t \
        No cream added: {t3:.2f}s")

#Begin solving problem
t = np.arange(0, 600., 0.5)
temp1 = solve_temp(t)           # also same as control case
temp2 = solve_temp(t, T0 = 85)  # cream added immediately case

fig, ax = plt.subplots(1,1)
ax.plot(t, temp1, label=f'Add Cream Later (T={t1:.0f}s)')
ax.plot(t, temp2, label=f'Add Cream Now (T={t2:.0f}s)')

ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (C)')
ax.set_title('When to add cream: Getting coffee cooled quickly')
fig.show()
#print(tfin)
