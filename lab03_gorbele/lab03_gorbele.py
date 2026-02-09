#!/usr/bin/env python3

'''
This file solves the diffusion equation for Lab 03 and all subparts.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
Ensure that any libraries imported below are installed to the user's
instance of python3.
In a terminal window opened to the directory (folder) that has this script
and is running python, enter `$ run lab03_gorbele.py`. The $ sign is not typed
by the user, but indicates where shell input begins.

Plots and results can be obtained by running the commands:
`$ verify_heatsolve()`, which addresses science question #1.
`$ kanger_diffusion()`, which addresses science question #2, and
`$ kanger_gw_diffusion()`, which adresses science question #3.
Plots are saved with descriptive names and are labelled.
To show the plots from the terminal, after running one of these commands,
the user can run `$ plt.show()` to display the current plot.

Note that, until the plot is closed or unless interactive mode is turned on,
the terminal will be occupied running the plot and will not
be able to run any additional functions.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

# OPTIMIZATIONS TO BE MADE:
# Code solver to accept input boundary/initial functions and also default to
# certain conditions without input to reduce redundant functions.


def solve_heat(func_0, func_ub, func_lb, xstop=100., tstop=150*365., dx=1,
               dt=0.1, c2=(0.25*10**-6)*86400, n=False, **kwargs):
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
    dx, dt : floats, defaults = 1m, 0.1 days
        Space and time step, respectively
    xstop, tstop : floats, defaults = 100m, 18,250 days (50 years)
        Length of object and time under consideration
    c2 : float, default = 0.25 mm^2*s^-1 (0.0216 m^2*days^-1)
        c^2 value for heat diffusivity
    n : bool, default = False
        True if Neumann boundary conditions, false if Dirichlet
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
    U = np.zeros([M, N])
    U[:, 0] = func_0(x, **kwargs)
    # print(U[:, 0])

    # Set upper boundary conditions across time
    U[0, :] = func_ub(t, **kwargs)
    U[-1, :] = func_lb(t, **kwargs)

    # Get r coeff:
    r = c2 * (dt/dx**2)

    # Solve heat equation
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])
        # Use Neumann conditions as applicable
        if n:
            U[0, j+1] = U[1, j+1]
            U[M-1, j+1] = U[M-2, j+1]

    # Return time and position vectors, and temperature array
    return t, x, U


# ---------------- Provide initial conditions for Rod problem ----------------

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
    t : numpy array
        The time vector.

    Returns
    -------
    ub : numpy array of zeros
        The upper boundary condition of the rod.
    '''
    ub = np.zeros(t.size)
    return ub


def verify_lbf(t):
    '''
    Takes time as an input and returns the lower boundary for the verification
    test of the heat solver function, constant 0 degrees Celsius.

    Parameters
    ----------
    t : numpy array
        The time vector.

    Returns
    -------
    lb : numpy array of zeros
        The lower boundary condition of the rod.
    '''
    lb = np.zeros(t.size)
    return lb


# --------------------------- Rod Problem Verifier ----------------------------

def verify_heatsolve(thresh=1E-6, **kwargs):
    '''
    Plots and prints example solution for 1-D rod diffusion equation against
    solver solution to verify solver function.

    Parameters
    ----------
    thresh : float, defaults to 0.01
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
                     [0.96, 0.80, 0.64, 0.52, 0.42, 0.34, 0.275, 0.2225,
                      0.18000, 0.145625, 0.117813],
                     [0.96, 0.80, 0.64, 0.52, 0.42, 0.34, 0.275, 0.2225,
                      0.18000, 0.145625, 0.117813],
                     [0.64, 0.48, 0.40, 0.32, 0.26, 0.21, 0.17, 0.1375,
                      0.11125, 0.090000, 0.072812],
                     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0000,
                      0.00000, 0.000000, 0.000000]])

    # Exit if on mismatches
    if (np.shape(u_ex) != np.shape(u)):
        raise ValueError("Array size mismatch with solver array.")

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

    # Check if example solution agrees with solver solution
    print('Is there disagreement between solver and example solution?')
    print(np.abs(u-u_ex).max() > thresh)

    return fig, ax, cbar, cbar_ex


# ------ Provide initial conditions for Kangerlussuaq Permafrost problem ------

# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
                    10.7, 8.5, 3.1, -6.0, -12.0, -16.9])


def temp_kangerub(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.

    Parameters
    ----------
    t : numpy array
        The time vector.

    Returns
    -------
    1-D numpy array of surface boundary conditions.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/182 * t - np.pi/2) + t_kanger.mean()


def temp_kangerlb(t):
    '''
    For an array of times in days, return timeseries of temperature for 100m
    depth under Kangerlussuaq, Greenland in degrees Celsius.

    Parameters
    ----------
    t : numpy array
        The time vector.

    Returns
    -------
    lb : 1-D numpy array
        Geothermal boundary conditions.
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
        Discretization of length of 1m wire.

    Returns
    -------
    U_0 : 1-D numpy array
        Initial temperatures across the vertical profile.
    '''
    U_0 = np.zeros(x.size)

    # Function input required for solver, but solver works to reach eq.
    # Return vector of zeros, boundary conditions to be overwritten later.
    return U_0


# ---------------------- Kangerlassuaq Permafrost Solver ----------------------

def kanger_diffusion(dt=0.1, tstop=150*365):
    '''
    Plots 1-D temperature profile from surface to 100m depth temperatures at
    Kangerlassuaq, Greenland over time and between seasons at equilibrium.
    Plot illustrates permafrost depth and temperature and structure
    over time given Dirichlet boundary conditions as it reaches equilibrium.

    Parameters
    ----------
    dt : float, defaults to 0.1 days
        Time step passed to heat solver.
    tstop : int, defaults to 150 years
        Stopping time passed to heat solver.

    Returns
    -------
    fig, (ax1, ax2) : matplotlib figure and axes objects
        The figure and axes of the plot.
    cbar : matplotlib color bar object
        The color bar on the final plot.
    '''

    # Get solution using solver and defaults:
    time, x, U = solve_heat(func_0=temp_kanger0, func_ub=temp_kangerub,
                            func_lb=temp_kangerlb, tstop=tstop, dt=dt)

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

    # Create a temp profile plot with a vertical line for ref:
    ax2.vlines(0, 0, 100, linestyles='dashed', color='gray')
    ax2.plot(winter, x, 'b--', label='Winter')
    ax2.plot(summer, x, 'r--', label='Summer')

    # Create yearly labels for comprehension; make a label every 25 years
    years = time[::int(25*365/dt)]/365
    ax1.set_xticks(time[::int(25*365/dt)], labels=years)

    # Axes labels and titles to be informative, include legend for seasons
    ax1.set_title('Ground Temperature Profile by Depth and Time')
    ax1.set_ylabel('Depth ($m$)')
    ax1.set_xlabel('Time ($Years$)')
    ax2.set_title(f'Ground Temperature Profile by Depth and Season, Year {int(tstop/365)}')
    ax2.set_ylabel('Depth ($m$)')
    ax2.set_xlabel(r'Temperature ($^{\circ}C$)')
    ax2.legend(loc='best')

    # Make figure more readable
    fig.tight_layout()

    # Active layer depth is defined as the layer that can reach above 0 Celsius
    active_layer = x[summer <= 0][0]
    # Use active layer depth as permafrost upper boundary
    # and define lower boundary as lowest point at/below 0 Celsius.
    pf_lower = x[summer <= 0][-1]

    print("Permafrost conditions:")
    print(f"Active layer depth at {int(tstop/365)} years: {active_layer} m")
    print(f"Permafrost depth at {int(tstop/365)} years: {pf_lower} m")

    return fig, (ax1, ax2), cbar


# ----- Provide initial conditions for Global Warming Permafrost problem ------

def temp_kangerub_05(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland under 0.5 degrees Celsius warming conditions.

    Parameters
    ----------
    t : numpy array
        The time vector.

    Returns
    -------
    1-D numpy array of surface boundary conditions under 0.5 degrees C warming.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/182 * t - np.pi/2) + t_kanger.mean() + 0.5


def temp_kangerub_10(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland under 1.0 degree Celsius warming conditions.

    Parameters
    ----------
    t : numpy array
        The time vector.

    Returns
    -------
    1-D numpy array of surface boundary conditions under 1.0 degrees C warming.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/182 * t - np.pi/2) + t_kanger.mean() + 1.0


def temp_kangerub_30(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland under 3.0 degrees Celsius warming conditions.

    Parameters
    ----------
    t : numpy array
        The time vector.

    Returns
    -------
    1-D numpy array of surface boundary conditions under 3.0 degrees C warming.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/182 * t - np.pi/2) + t_kanger.mean() + 3.0


# --------------------- Global Warming Permafrost Solver ----------------------

def kanger_gw_diffusion(tstop=75*365, dt=0.1):
    '''
    Plots 1-D equilibrium temperature profile from surface to 100m depth at
    Kangerlassuaq, Greenland time under different global warming conditions.

    Parameters
    ----------
    tstop : float, defaults to 50 years (converted to days)
        the amt of time to call thing reuse from before
    dt : float, defaults to 0.1 days

    Returns
    -------
    fig, (ax1, ax2) : matplotlib figure and axes objects
        The figure and axes of the plot.
    '''

    # Create figure and axes objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # explode the universe: run the three situations at necessary equilbirum
    # reaching times and then use the 2nd part of the plot w/ winter/summer
    # conditions/structure and finish that stuff up!!

    # Allowed to use t and x again because they SHOULDNT change
    t05, x05, U05 = solve_heat(func_0=temp_kanger0, func_ub=temp_kangerub_05,
                               func_lb=temp_kangerlb, tstop=tstop, dt=dt)
    t10, x10, U10 = solve_heat(func_0=temp_kanger0, func_ub=temp_kangerub_10,
                               func_lb=temp_kangerlb, tstop=tstop, dt=dt)
    t30, x30, U30 = solve_heat(func_0=temp_kanger0, func_ub=temp_kangerub_30,
                               func_lb=temp_kangerlb, tstop=tstop, dt=dt)

    # Do some valueerror raises to see if different sizes or whatever
    # Can then allocate x to simply x05 since all are same
    if ((np.shape(U05) != np.shape(U10)) | (np.shape(U05) != np.shape(U30))):
        raise ValueError("Array sizes are mismatched.")

    # Allocate common x variable
    x = x05

    # Set indexing for the final year of results:
    loc = int(-365/dt)

    # Extract the min and max values over the final year per season:
    winter05 = U05[:, loc:].min(axis=1)
    summer05 = U05[:, loc:].max(axis=1)
    winter10 = U10[:, loc:].min(axis=1)
    summer10 = U10[:, loc:].max(axis=1)
    winter30 = U30[:, loc:].min(axis=1)
    summer30 = U30[:, loc:].max(axis=1)

    # Plot the profiles on their separate plots
    ax2.plot(summer05, x, color='sienna',      label=r'+ 0.5 $^{\circ}C$')
    ax1.plot(winter05, x, color='sienna',      label=r'+ 0.5 $^{\circ}C$')

    ax2.plot(summer10, x, color='orange',      label=r'+ 1.0 $^{\circ}C$')
    ax1.plot(winter10, x, color='orange',      label=r'+ 1.0 $^{\circ}C$')

    ax2.plot(summer30, x, color='deepskyblue', label=r'+ 3.0 $^{\circ}C$')
    ax1.plot(winter30, x, color='deepskyblue', label=r'+ 3.0 $^{\circ}C$')

    # Provide labels for the axes and a vertical line to indicate 0 degrees
    for ax in (ax1, ax2):
        ax.set_xlabel(r'Temperature ($^{\circ}C$)')
        ax.set_ylabel('Depth ($m$)')
        ax.vlines(0, 0, 100, linestyles='dashed', color='gray')
        ax.yaxis.set_inverted(True)

    # Provide legends and titles for interpretation
    ax2.set_title(f'Summer')
    ax1.set_title(f'Winter')
    fig.suptitle(f"Ground Temperature Profile\nby Depth Under Warming Regimes, year {int(tstop/365)}")
    ax1.legend(loc='best')
    ax2.legend(loc='lower right')

    fig.tight_layout()

    # Active layer depth is defined as the layer that can reach above 0 Celsius
    active_layer05 = x[summer05 <= 0][0]
    active_layer10 = x[summer10 <= 0][0]
    active_layer30 = x[summer30 <= 0][0]
    # Use active layer depth as permafrost upper boundary
    # and define lower boundary as lowest point at/below 0 Celsius.
    pf_lower05 = x[summer05 <= 0][-1]
    pf_lower10 = x[summer10 <= 0][-1]
    pf_lower30 = x[summer30 <= 0][-1]

    print("Permafrost conditions for 0.5 degrees C warming:")
    print(f"Active layer depth at {int(tstop/365)} years: {active_layer05} m")
    print(f"Permafrost depth at {int(tstop/365)} years: {pf_lower05} m\n")

    print("Permafrost conditions for 1.0 degrees C warming:")
    print(f"Active layer depth at {int(tstop/365)} years: {active_layer10} m")
    print(f"Permafrost depth at {int(tstop/365)} years: {pf_lower10} m\n")

    print("Permafrost conditions for 3.0 degrees C warming:")
    print(f"Active layer depth at {int(tstop/365)} years: {active_layer30} m")
    print(f"Permafrost depth at {int(tstop/365)} years: {pf_lower30} m")

    return fig, ax
