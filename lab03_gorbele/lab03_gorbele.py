#!/usr/bin/env python3

'''
This file solves the diffusion equation for Lab 03 and all subparts.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


def solve_heat(func_0, func_ub, func_lb, xstop=100, tstop=50, dx=0.1, 
               dt=1/365., c2=0.25*10**-6, d=False, **kwargs):
    '''
    Solves the 1-dimensional diffusion equation.

    Parameters
    ----------
    func_0 : function
        A python function that takes `x` as input and returns the initial 
        conditions of the diffusion example under consideration.
    func_ub : function, default = 0
        A python function that takes `t` as input and returns the upper
        conditions of the diffusion example under consideration.
    func_lb : function
        A python function that takes `t` as input and returns the lower
        conditions of the diffusion example under consideration.
    dx, dt : floats, defaults = 0.1m, 1/365. years (1 day)
        Space and time step, respectively
    xstop, tstop : floats, defaults = 100m, 50 years
        Length of object and time under consideration
    c2 : float, default = 0.25 mm^2*s^-1
        c^2 value for heat diffusivity
    d : bool, default = False
        True if Dirichlet boundary conditions, false if naumann
    **kwargs : keyword arguments
    
    Returns
    -------
    x, t : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    '''

    # Check our stability criterion, exit if unstable
    dtmax = dx**2 / (2*c2)
    if dt > dtmax:
        raise ValueError(f"Stablility criterion not met, dt={dt} > dt_max={dtmax}.")

    # Get grid sizes:
    N = int(np.floor(tstop / dt)) + 1
    M = int(np.floor(xstop / dx)) + 1

    
    # Set up space and time grid:
    x = np.linspace(0, xstop, M)
    t = np.linspace(0, tstop, N)

    # Create solution matrix; set init conditions
    U = np.zeros([M,N])
    U[:,0] = func_0(x, **kwargs)
    print(U[:,0])

    # Set upper boundary conditions across time
    U[0,:] = func_ub(t, **kwargs)
    U[-1,:] = func_lb(t, **kwargs)

    # Get r coeff:
    r = c2 * (dt/dx**2)

    # Solve heat equation
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])
        # Use Dirichlet conditions as applicable
        if d: 
            U[0, j+1] = U[1, j+1]
            U[M-1, j+1] = U[M-2, j+1]


    # Return time and position vectors and temperature array
    return t, x, U

def verify_initf(x):
    '''
    This function calculates the initial temperature conditions across the 
    vertical profile of the environment for the example of the 1m wire 

    Parameters
    ----------
    x : 1-D numpy array
        Discretization of length of 1m wire.

    Returns
    -------
    U_0 : 1-D numpy array
        Initial temperatures across the 1m wire.
    '''

    U_0 = 4*x - x**4 # initial temperature distribution in degrees Celsius

    return U_0

def verify_ubf(t):
    '''
    Takes time as an input and returns the upper boundary for the verification
    test of the heat solver function, constant 0 degrees Celsius.
    '''
    ub = np.zeros(t)
    return ub

def verify_lbf(t):
    '''
    Takes time as an input and returns the lower boundary for the verification
    test of the heat solver function, constant 0 degrees Celsius.
    '''
    lb = np.zeros(t)
    return lb

def verify_heatsolvet(**kwargs):
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

    # Solve for heat equation using initial conditions
    t, x, u = solve_heat(verify_initf,func_ub=0, func_lb=0,
                         xstop=1,dx=0.2,tstop=0.2,dt=0.02,c2=1)

    # Add contour to axis:
    contour = ax.pcolor(t, x, u, **kwargs)
    cbar = plt.colorbar(contour)

    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Position ($m$)')
    ax.set_title('Dirichlet Boundary Conditions Diffusion Example')

    fig.tight_layout()

    return fig, ax, cbar

# checks if numerically stable- stability criterion already implemented :3