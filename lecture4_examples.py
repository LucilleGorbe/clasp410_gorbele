#!/usr/bin/env python3

'''
Diffusion. woww. send me to the abyss please
Tools and methods for completing lab 3 which is arguably the best lab maybe
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

def solve_heat(xstop=1, tstop=0.2, dx=0.02, dt=0.0002, c2=1, d=False):
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
    d : bool
        True if dirichlet boundary conditions, false if naumann
    
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
    print(U[:,0])

    # get r coeff:
    r = c2 * (dt/dx**2)

    #solve da eq
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])
        if d: 
            U[0, j+1] = U[1, j+1]
            U[M-1, j+1] = U[M-2, j+1]


    # return to ME@E@E#E!E*E&E^E^E^%%E$%W^E@#$%^&@!*
    return t, x, U

def plot_heatsolvet(**kwargs):
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
    fig, ax = plt.subplots(2,1, figsize=(8,8))

     # Check kwargs for defaults
    if 'cmap' not in kwargs:
        kwargs['cmap']='hot'

    tn, xn, un = solve_heat()
    td, xd, ud = solve_heat(d=True)

    # Add contour to axis:
    contourn = ax[0].pcolor(tn, xn, un, **kwargs)
    contourd = ax[1].pcolor(td, xd, ud, **kwargs)
    cbarn = plt.colorbar(contourn)
    cbard = plt.colorbar(contourd)

    cbarn.set_label(r'Temperature ($^{\circ}C$)')
    cbard.set_label(r'Temperature ($^{\circ}C$)')
    ax[0].set_xlabel('Time ($s$)')
    ax[0].set_ylabel('Position ($m$)')
    ax[1].set_xlabel('Time ($s$)')
    ax[1].set_ylabel('Position ($m$)')
    ax[0].set_title('Dirichlet Boundary Conditions Diffusion Example')
    ax[1].set_title('Neumann Boundary Conditions Diffusion Example')

    fig.tight_layout()

    return fig, ax, cbarn, cbard