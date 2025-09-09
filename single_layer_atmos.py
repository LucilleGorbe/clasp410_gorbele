#!/usr/bin/env python3

'''
This script contains a function for exploring temperature under a single-layer atmosphere model.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

# Bring those constants out of constant jail
sig = 5.67E-8 #Steffan Boltzmann const.

# values needed for solar forcing exploration
year = np.array([1900, 1950, 2000])
s0 = np.array([1365.0, 1366.5, 1368.0]) #solar forcing in W/m^2
T_anom = np.array([-.4, 0., 0.4]) # temp anomaly since 1950 in K

#function to give surface temperature as a funct of solar forcing and albedo
# T_anom = Tnow - T_1950 ; so Tnow = T_anom + T_1950

def surfT_1layer(s0=1350, albedo=0.33):

    '''
    Produces a surface temperature based on input solar forcings under an energy-balanced,
    perfectly-absorbing, 1-layer atmospheric model.
    ----------
    Parameters
    ----------
    s0 - float
        Incoming solar flux (in W/m^2) of the Earth.
    albedo - float
        Albedo value of Earth.
    ----------
    Returns
    ----------
    T - float
        Surface temperature (in K) as predicted by the energy balance equation.
    '''

    T=((1-albedo)*s0/(2*sig))**(1/4)
    return T
    
# call function to get predicted temperature by solar forcings
T_solar = surfT_1layer(s0=s0)
# We're analyzing the difference in effect of measured anomaly vs predicted anomaly, 
# so we can use the 1950 value for solar forcing temp and simply add the anomaly for later comparison
T_measured = T_solar[1] + T_anom

#Create a plot with our two separate lines (w/ labels for differentiating) and visible line widths
fig, ax = plt.subplots(1,1, figsize = (8,5))
ax.plot(year, T_solar, label='Solar Forcing Predicted Temperature', lw=3)
ax.plot(year, T_measured, label='Measured Temperature', lw=3)

#labelling for all those awesome DanPoints
ax.legend(loc='best')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature ($K$)')
ax.set_title('Solar Forcing Changes Do Not Explain Measured Earth Temperature Changes')

