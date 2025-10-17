#!/usr/bin/env python3

'''
Diffusion. woww. send me to the abyss please
Tools and methods for completing lab 3 which is arguably the best lab maybe
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

def solve_heat(xstop=1, tstop=0.2, dx=0.02, dt=0.0002, c2=1):
    '''
    Me when i solve the head equation for a rod.

    Parameters
    ----------
    dx, dt : floats
        Space and time step, respectively
    xstop, tstop : floats
        size of rod and length of time under consideration
    c2 : float
        c^2 value for heat diffusivity
    
    Returns
    -------
    x, t : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    
    '''

    # Check our stability criterion 
    dtmax = dx**2 / (2*c2)
    if dt > dtmax:
        raise ValueError(f"Danger!!!!!!!!!! dt={dt} > dt_max={dtmax}, stablility criterion not met. Fuck you")

    # Get grid sizes:
    N = int(np.floor(tstop / dt)) + 1
    M = int(np.floor(xstop / dx)) + 1

    
    # Set up space and time grid:
    x = np.linspace(0, xstop, M)
    t = np.linspace(0, tstop, N)

    # Create solution matrix; set init conditions
    U = np.zeros([M,N])
    U[:,0] = 4*x - 4*x**2

    # get r coeff:
    r = c2 * (dt/dx**2)

    #solve da eq
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])
        U[0, j+1] = U[1, j+1]
        U[M-1, j+1] = U[M-2, j+1]


    # return to ME@E@E#E!E*E&E^E^E^%%E$%W^E@#$%^&@!*
    return t, x, U

def plot_heatsolvet(t,x,U,title=None, **kwargs):
    '''
    Plot 2d solution for solve_heat function.

    Parameters
    ----------
    **kwargs : keyword arguments

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
        The figure and axes of the plot.
    cbar : matplotlib color bar object
        The color bar on the final plot
    '''

    # Create and configure figure & axis
    fig, ax = plt.subplots(1,1, figsize=(8,8))

     # Check kwargs for defaults
    if 'cmap' not in kwargs:
        kwargs['cmap']='hot'

    # Add contour to axis:
    contour = ax.pcolor(t, x, U, **kwargs)
    cbar = plt.colorbar(contour)

   
        

    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Position ($m$)')
    ax.set_title(title)

    return fig, ax, cbar