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


def solve_heat(func_0, func_ub, func_lb, xstop=100., tstop=50*365., dx=1,
               dt=0.1, c2=(0.25*10**-6)*86400, d=False, **kwargs):
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
    dx, dt : floats, defaults = 0.1m, 0.1 days
        Space and time step, respectively
    xstop, tstop : floats, defaults = 100m, 18,250 days (50 years)
        Length of object and time under consideration
    c2 : float, default = 0.25 mm^2*s^-1 (0.0216 m^2*days^-1)
        c^2 value for heat diffusivity
    d : bool, default = False
        True if Dirichlet boundary conditions, false if naumann
    **kwargs : keyword arguments

    Returns
    -------
    t, x : 1D Numpy arrays
        Time and space values, respectively.
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
    U = np.zeros([M, N])
    U[:, 0] = func_0(x, **kwargs)
    print(U[:, 0])

    # Set upper boundary conditions across time
    U[0, :] = func_ub(t, **kwargs)
    U[-1, :] = func_lb(t, **kwargs)

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
    vertical profile of the environment for the example of the 1m wire.

    Parameters
    ----------
    x : 1-D numpy array
        Discretization of length of 1m wire.

    Returns
    -------
    U_0 : 1-D numpy array
        Initial temperatures across the 1m wire.
    '''

    U_0 = 4*x - 4*x**2  # Initial temperature distribution in degrees Celsius

    return U_0


def verify_ubf(t):
    '''
    Takes time as an input and returns the upper boundary for the verification
    test of the heat solver function, constant 0 degrees Celsius.
    
    Parameters
    ----------
    t : 1-D numpy array
        Discretization of time of the simulation.

    Returns
    -------
    ub : 1-D numpy array
        Upper boundary temperature, always 0°C .
    '''
    ub = np.zeros(t.size)
    return ub


def verify_lbf(t):
    '''
    Takes time as an input and returns the lower boundary for the verification
    test of the heat solver function, constant 0 degrees Celsius.
    
    Parameters
    ----------
    t : 1-D numpy array
        Discretization of time of the simulation.

    Returns
    -------
    lb : 1-D numpy array
        Lower boundary temperature, always 0°C .
    '''
    lb = np.zeros(t.size)
    return lb


def verify_heatsolvet(thresh=1E-6, **kwargs):
    '''
    Plots and prints example solution for 1-D rod diffusion equation against
    solver solution to verify solver function.

    Parameters
    ----------
    thresh : float
        Threshold for floating point differences when comparing solver
        solution and example solution.
    **kwargs : keyword arguments

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
        The figure and axes of the plot.
    cbar, cbar_ex : matplotlib color bar objects
        The color bars on the final plot for solver and example, respectively.
    '''

    # Create and configure figure & axes
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    # Check kwargs for defaults
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'

    # Solve for heat equation using initial conditions
    t, x, u = solve_heat(verify_initf, verify_ubf, verify_lbf,
                         xstop=1, dx=0.2, tstop=0.2, dt=0.02, c2=1)

    # Example solution for comparison
    u_ex = np.array([[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0000,
                      0.00000, 0.000000, 0.000000],
                     [0.64, 0.48, 0.40, 0.32, 0.26, 0.21, 0.17, 0.1375,
                      0.11125, 0.090000, 0.072812],
                     [0.96, 0.80, 0.64, 0.52, 0.42, 0.34, 0.28, 0.2225,
                      0.18000, 0.145625, 0.117813],
                     [0.96, 0.80, 0.64, 0.52, 0.42, 0.34, 0.28, 0.2225,
                      0.18000, 0.145625, 0.117813],
                     [0.64, 0.48, 0.40, 0.32, 0.26, 0.21, 0.17, 0.1375,
                      0.11125, 0.090000, 0.072812],
                     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0000,
                      0.00000, 0.000000, 0.000000]])

    # To Do's: add errors exits based on shape of arrays, compare absolute value
    #if u_ex.shape() != u.shape():
    #    raise ValueError("Array size mismatch with solver array.")
    #if (x.shape()[0], t.shape()[0]) != u_ex.shape():
    #    raise ValueError("Array size mismatch with input vectors.")

    # Add contour to axis:
    contour = ax[0].pcolor(t, x, u, **kwargs)
    cbar = plt.colorbar(contour)
    contour_ex = ax[1].pcolor(t, x, u_ex, **kwargs)
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
    print(f'\t{np.abs(u-u_ex).max() > thresh}')

    return fig, ax, cbar, cbar_ex


# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
                    10.7, 8.5, 3.1, -6.0, -12.0, -16.9])


def temp_kangerub(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/182 * t - np.pi/2) + t_kanger.mean()


def temp_kangerlb(t):
    '''
    For an array of times in days, return timeseries of temperature for 100m
    depth under Kangerlussuaq, Greenland in degrees Celsius.
    '''
    lb = np.zeros(t.size)
    lb[:] = 5
    return lb


def temp_kanger0(x):
    '''
    This function returns the initial temperature conditions across the
    vertical profile of the environment at Kangerlussuaq, Greenland in degrees
    Celsius.

    Parameters
    ----------
    x : 1-D numpy array
        Discretization of the soil depth.

    Returns
    -------
    U_0 : 1-D numpy array
        Initial temperatures across the vertical profile.
    '''
    U_0 = np.zeros(x.size)

    # Function input required for solver, but solver works to reach eq.
    # Return vector of zeros, boundary conditions to be overwritten later.
    return U_0


def kanger_diffusion():
    '''
    Plots 1-D temperature profile from surface to 100m depth temperatures at
    Kangerlassuaq, Greenland over time. Plot illustrates permafrost depth and
    temperature and structure over time given Neumann boundary conditions
    as it reaches equilibrium.

    Returns
    -------
    fig, (ax1, ax2) : matplotlib figure and axes objects
        The figure and axes of the plot.
    cbar : matplotlib color bar object
        The color bar on the final plot.
    '''

    dt = 0.2
    dx = 1
    tstop = 150*365

    # Get solution using solver and defaults:
    time, x, U = solve_heat(func_0=temp_kanger0, func_ub=temp_kangerub,
                            func_lb=temp_kangerlb, tstop=tstop, dx=dx, dt=dt)

    # Create a figure/axes object
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Reverse axes to match depth profile standards
    ax1.yaxis.set_inverted(True)
    ax2.yaxis.set_inverted(True)

    # Create a color map and add a color bar.
    profile = ax1.pcolor(time, x, U, cmap='seismic', vmin=-25, vmax=25)
    cbar = plt.colorbar(profile, ax=ax1, label=r'Temperature ($^{\circ}C$)')

    # Set indexing for the final year of results:
    loc = int(-365/dt)

    # Extract the min and max values over the final year per season:
    winter = U[:, loc:].min(axis=1)
    summer = U[:, loc:].max(axis=1)

    # Create a temp profile plot:
    ax2.plot(winter, x, 'b--', label='Winter')
    ax2.plot(summer, x, 'r--', label='Summer')

    # Label axes to be informative
    ax1.set_title('Ground Temperature Profile by Depth and Time')
    ax2.set_title('Ground Temperature Profile by Depth and Season')
    ax2.set_ylabel('Depth ($m$)')
    ax2.set_xlabel(r'Temperature ($^{\circ}C$)')
    ax2.legend(loc='best')

    return fig, (ax1, ax2), cbar


def temp_kangerub_05(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland under 0.5 degrees Celsius warming conditions.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/182 * t - np.pi/2) + t_kanger.mean() + 0.5


def temp_kangerub_10(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland under 1.0 degree Celsius warming conditions.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/182 * t - np.pi/2) + t_kanger.mean() + 1.0


def temp_kangerub_30(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland under 3.0 degrees Celsius warming conditions.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/182 * t - np.pi/2) + t_kanger.mean() + 3.0


def kanger_gw_diffusion():
    '''
    This function...... under global warming conditions.

    Returns
    -------
    fig, (ax1, ax2, ax3) : matplotlib figure and axes objects
        The figure and axes of the plot.
    '''

    # Create figure and axes objects
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # explode the universe: run the three situations at necessary equilbirum 
    # reaching times and then use the 2nd part of the plot w/ winter/summer 
    # conditions/structure and finish that stuff up!!


    return fig, (ax1, ax2, ax3)
