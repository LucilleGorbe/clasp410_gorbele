#!/usr/bin/env python3

'''
This file solves the diffusion equation for Lab 03 and all subparts.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

# OPTIMIZATIONS TO BE MADE:
# Code solver to accept input boundary/initial functions and also default to 
# certain conditions without input to reduce redundant functions.

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
    Plots and prints example solution for 1-D rod diffusion equation against 
    solver solution to verify solver function.

    Parameters
    ----------
    **kwargs : keyword arguments

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
        The figure and axes of the plot.
    cbar, cbar_ex : matplotlib color bar objects
        The color bars on the final plot for solver and example, respectively.
    '''

    # Create and configure figure & axes
    fig, ax = plt.subplots(2,1, figsize=(8,8))

    # Check kwargs for defaults
    if 'cmap' not in kwargs:
        kwargs['cmap']='hot'

    # Solve for heat equation using initial conditions
    t, x, u = solve_heat(verify_initf,verify_ubf,verify_lbf,
                         xstop=1,dx=0.2,tstop=0.2,dt=0.02,c2=1)
    
    # Comparison solution 
    u_ex = np.array([[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0000, 
                      0.00000, 0.000000, 0.000000]
                     [0.64, 0.48, 0.40, 0.32, 0.26, 0.21, 0.17, 0.1375, 
                        0.11125, 0.090000, 0.072812]
                     [0.96, 0.80, 0.64, 0.52, 0.42, 0.34, 0.28, 0.2225, 
                        0.18000, 0.145625, 0.117813]
                     [0.96, 0.80, 0.64, 0.52, 0.42, 0.34, 0.28, 0.2225, 
                        0.18000, 0.145625, 0.117813]
                     [0.64, 0.48, 0.40, 0.32, 0.26, 0.21, 0.17, 0.1375, 
                        0.11125, 0.090000, 0.072812]
                     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0000, 
                        0.00000, 0.000000, 0.000000]])

    # Add contour to axis:
    contour = ax.pcolor(t, x, u, **kwargs)
    cbar = plt.colorbar(contour)
    contour_ex = ax.pcolor(t, x, u_ex, **kwargs)
    cbar_ex = plt.colorbar(contour_ex)

    # Label plots for readability
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax[0].set_xlabel('Time ($s$)')
    ax[0].set_ylabel('Position ($m$)')
    ax[0].set_title('Rod Diffusion Verification Solver Solution')
    cbar_ex.set_label(r'Temperature ($^{\circ}C$)')
    ax[1].set_xlabel('Time ($s$)')
    ax[1].set_ylabel('Position ($m$)')
    ax[1].set_title('Rod Diffusion Verification Example Solution')   

    fig.tight_layout()

    # Print solver and example differences 
    print('Disagreement between solver and example solution:')
    print(u-u_ex)

    return fig, ax, cbar, cbar_ex

# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
                    10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

def temp_kangerup(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()

def temp_kangerlb(t):
    '''
    For an array of times in days, return timeseries of temperature for 100m
    depth under Kangerlussuaq, Greenland.
    '''
    lb = np.zeros(t)
    lb[:] = 5 # degrees Celsius
    return lb

def temp_kanger0(x):
    '''
    This function returns the initial temperature conditions across the 
    vertical profile of the environment at Kangerlussuaq, Greenland.
    
    Parameters
    ----------
    x : 1-D numpy array
        Discretization of length of 1m wire.

    Returns
    -------
    U_0 : 1-D numpy array
        Initial temperatures across the vertical profile.
    '''
    U_0 = np.zeros(x)

    # Function input required for solver, but solver works to reach eq.
    # Return vector of zeros, boundary conditions to be overwritten later.
    return U_0

def kanger_diffusion():
    '''
    Plots 1-D diffusion 

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
        The figure and axes of the plot.
    cbar : matplotlib color bar object
        The color bar on the final plot
    '''

    # Get solution using your solver:
    time, x, U = solve_heat()

    # Create a figure/axes object
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))

    # Create a color map and add a color bar.
    map = ax1.pcolor(time, x, U, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=ax1, label='Temperature ($C$)')



    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result.

    # Extract the min values over the final year:
    winter = U[:, loc:].min(axis=1)

    # Create a temp profile plot:
    ax2.plot(winter, x, label='Winter')