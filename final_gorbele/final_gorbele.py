'''
Mi final project D:
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


def solve_conc(func_0, func_lb, xstop=10., tstop=60., dx=0.1,
               dt=0.1, D=11.6E-10, v=0.1, **kwargs):
    '''
    Solves the 1-dimensional advection-diffusion equation.

    Parameters
    ----------
    func_0 : function
        A python function that takes `x` as input and returns the initial
        conditions of the diffusion example under consideration.
    func_lb : float, vector, or function
        A python function that takes `t` as input and returns the constant
        lower boundary conditions (i.e. the material concentration inflow rate)
        of the advection-diffusion example under consideration.
    dx, dt : floats, defaults = 0.1m, 0.1 seconds
        Space and time step, respectively
    xstop, tstop : floats, defaults = 10m, 20 seconds
        Length of object and time under consideration
    D : float, default = 11.6E-10
        Diffusivity coefficient in m^2/s. Default value adapted from arsenic in
        water from Tanaka, Takahashi, et al. (2013).
    v : float, default = 0.1
        Velocity in m/s
    n : bool, default = False
        True if Neumann boundary conditions, false if Dirichlet
    **kwargs : keyword arguments

    Returns
    -------
    x, t : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the advection-diffusion equation, 
        size is nSpace x nTime
    '''

    # Check our stability criterion, exit if unstable
    # Will potentially need to change due to intro of 1st order error
    dtmaxdiff = dx**2 / (2*D)
    dtmaxadv = dx / (2*v)
    dtmax = dtmaxdiff * (dtmaxdiff >= dtmaxadv) + dtmaxadv * (dtmaxdiff < dtmaxadv)
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

    # Set lower boundary conditions across time
    U[0, :] = func_lb(t, **kwargs)  # LOWER BOUNDARY

    # Get r and p coeffs:
    r = D * (dt/dx**2)
    p = v * (dt/dx)

    # Solve heat equation
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r-p) * U[1:M-1, j] + r*U[2:M, j] + (r+p)*U[:M-2, j]
        # Use Neumann condition for outflow boundary
        U[M-1, j+1] = U[M-2, j]

    # Return time and position vectors, and temperature array
    return t, x, U


def conc_riverlb(t):
    '''
    For an array of times in seconds, return timeseries of Buckeyelium
    concentration at the factory source in g/m^3. The factory has a constant
    Buckeyelium input into the river of 5g/m^3.

    Parameters
    ----------
    t : numpy array
        The time vector.

    Returns
    -------
    lb : 1-D numpy array
        Factory boundary conditions.
    '''
    lb = np.zeros(t.size)
    lb[:] = 5 * np.exp(-t/200)
    return lb


def conc_river0(x):
    '''
    This function returns the initial Buckeyelium concentrations across the
    horizontal profile of the environment at Awesome River in g/m^3

    Parameters
    ----------
    x : 1-D numpy array
        Discretization of length of 100m river.

    Returns
    -------
    U_0 : 1-D numpy array
        Initial concentration of Buckeyelium across the horizontal profile.
    '''
    U_0 = np.zeros(x.size)

    # Function input required for solver, but solver works to reach eq.
    # Return vector of zeros, boundary conditions to be overwritten later.
    return U_0


def buckeyelium(dt=0.01, tstop=60, v=0.5, D=0.002):
    '''
    New dangerous forever chemical called Buckeyeallium
    Crimson (sometimes red color), smells bad, horribly processed
    Used in making OSU brand sports drinks. They dump it in the river.

    Parameters
    ----------
    dt : float, defaults to 0.01 seconds
        Time step passed to concentration solver.
    tstop : int, defaults to 60 seconds
        Stopping time passed to concentration solver.
    v : float, defaults to 0.1 m/s
        Velocity of the stream passed to the concentration solver.

    Returns
    -------
    fig, (ax1, ax2) : matplotlib figure and axes objects
        The figure and axes of the plot.
    cbar : matplotlib color bar object
        The color bar on the final plot.
    '''

    # Get solution using solver and defaults:
    time, x, U = solve_conc(func_0=conc_river0, func_lb=conc_riverlb, dt=dt,
                            tstop=tstop, v=v, D=D)
    
    logU = np.log10(U)

    # Create a figure/axes object
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Create a color map and add a color bar.
    profile = ax2.pcolor(time, x, U, cmap='summer', vmin=0)
    cbar = plt.colorbar(profile, ax=ax2, label=r'Concentration ($\frac{g}{m^3}$)')

    ax1.plot(x, U[:, -1])
    ax1.set_title("Buckeyelium for final timestep")

    # Axes labels and titles to be informative, include legend for seasons
    ax2.set_xlabel('Time ($s$)')
    ax2.set_ylabel('Position ($m$)')
    ax2.set_title('Buckeyelium Concentration over Time')

    # Make figure more readable
    fig.tight_layout()

    return fig, (ax1, ax2), cbar


def no_adv_test():
    '''
    Test for no advection occurring to ensure that the diffusive properties
    of the solver function as intended
    '''

    # set input velocity to zero
    v = 0