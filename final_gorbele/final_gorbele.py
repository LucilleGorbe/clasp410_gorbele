'''
Final project of Lucille Gorbe for Climate 410.

To generate all solution plots, which will be saved in the folder that this
script is located in, DO THIS:
Ensure that any libraries imported below are installed to the user's
instance of python3.
In a terminal window opened to the directory (folder) that has this script
and is running python, enter `$ run makeup_gorbele.py`. The $ sign is not typed
by the user, but indicates where shell input begins.

Plots and results can be obtained by running the commands:
`$ get_all_plots()`, the solution plots for science questions #1, #2, and #3.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


def solve_conc(func_0, func_lb, xstop=50., tstop=200., dx=0.1, dt=0.01,
               D=11.6E-10, v=0.01, filter_net=False, **kwargs):
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
    xstop, tstop : floats, defaults = 50m, 200 seconds
        Length of object and time under consideration
    D : float, default = 11.6E-10
        Diffusivity coefficient in m^2/s. Default value adapted from arsenic in
        water from Tanaka, Takahashi, et al. (2013).
    v : float, default = 0.01 m/s
        velocity of the flow field. Default value adapted from measurements of
        Miller's Creek current during low flow.
    filter_net : boolean, default = False
        Application of filter to retain some material from outflow.
    **kwargs
        Keyword arguments

    Returns
    -------
    x, t : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the advection-diffusion equation, 
        size is nSpace x nTime
    '''

    # Because this solver is designed with 1-dimensional flow at the exits,
    # Return error if vmax is negative.
    if v < 0:
        raise ValueError(f"v must be a positive number or zero. input v={v}")

    # Check our stability criterion, exit if unstable
    dtmaxdiff = dx**2 / (2*D)
    # Use np.inf if vmax is set to 0, otherwise will cause errors and slowdowns
    if v == 0:
        dtmaxadv = np.inf
    else:
        dtmaxadv = dx / (2*v)
    # Choose the smaller dt max
    dtmax = dtmaxdiff * (dtmaxdiff <= dtmaxadv) + dtmaxadv * (dtmaxdiff > dtmaxadv)
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

    # If filter_net is defined, set 70% filter efficiency
    filter_efficiency = 0.0
    if filter_net:
        filter_efficiency = 0.7

    # Solve advection-diffusion equation
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r-p) * U[1:M-1, j] + r*U[2:M, j] + (r+p)*U[:M-2, j]
        # Use Neumann condition for outflow boundary 
        # AND IF FILTER_NET is True, keep capturing things
        U[M-1, j+1] = U[M-2, j] + (U[M-1, j] * filter_efficiency)


    # Return time and position vectors, temperature array
    return t, x, U


def conc_riverlb(t):
    '''
    For an array of times in seconds, return timeseries of Buckeyelium
    concentration at the source in g/m^3. The Buckeyelium input into 
    the river starts at 5mg/m^3 and exponentially decreases over time, 
    representing the accidental drainage of small buckeyelium-flavored drinks 
    (which OSU placed in the stream and pierced accidentally, they promise).

    Parameters
    ----------
    t : numpy array
        The time vector.

    Returns
    -------
    lb : 1-D numpy array
        Input boundary conditions.
    '''
    lb = np.zeros(t.size) + (5 * np.exp(0.005*-t))
    return lb


def conc_river0(x):
    '''
    This function returns the initial Buckeyelium concentrations across the
    profile of the 1-D stream in g/m^3, which is none. The input will be
    handled after this is set.

    Parameters
    ----------
    x : 1-D numpy array
        Discretization of length of the stream.

    Returns
    -------
    U_0 : 1-D numpy array
        Initial concentration of Buckeyelium across the horizontal profile.
    '''
    U_0 = np.zeros(x.size)

    # Function input required for solver, but solver works to reach eq.
    # Return vector of zeros, boundary conditions to be overwritten later.
    return U_0


def buckeyelium(xstop=10., tstop=1000., dx=0.1, dt=0.1, D=11.6E-10, v=0.01,
                filter_net=False):
    '''
    A new dangerous forever chemical called Buckeyeallium has infiltrated
    the Miller's Creek, er, a 1-dimensional representation of Miller's Creek.
    Crimson or red colored, buckeyelium smells bad and is toxic to creek
    ecology and humans, when consumed. Though, unfortuantely for all who live
    in Ohio, it's commonly used to make OSU brand sports drinks. 
    Buckeyelium enters an ideal 1-dimensional stream, and, in this case,
    with constant velocity at an exponentially decaying rate of substance 
    addition. 

    Parameters
    ----------
    dx, dt : floats, defaults = 0.1m, 0.1 seconds
        Space and time step, respectively
    xstop, tstop : floats, defaults = 50m, 200 seconds
        Length of object and time under consideration
    D : float, default = 11.6E-10
        Diffusivity coefficient in m^2/s. Default value adapted from arsenic in
        water from Tanaka, Takahashi, et al. (2013).
    v : float, default = 0.01 m/s
        velocity of the flow field. Default value adapted from measurements of
        Miller's Creek current during low flow.
    filter_net : boolean, default = False
        Application of filter to retain some buckeyelium from outflow.
    Returns
    -------
    fig : matplotlib figure object
        The figure plotted on for the advection-diffusion test
    '''

    # Get solution using solver and defaults:
    time, x, U = solve_conc(func_0=conc_river0, func_lb=conc_riverlb,
                                    dx=dx, xstop=xstop, dt=dt, tstop=tstop,
                                    D=D, v=v, filter_net=filter_net)

    # Create a figure/axes object
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Create a color map and add a color bar.
    profile = ax2.pcolor(time, x, U, cmap='summer', vmin=0)
    plt.colorbar(profile, ax=ax2, label=r'Concentration ($\frac{g}{m^3}$)')

    # Plot the final timestep of buckeyelium concentration
    ax1.plot(x, U[:, -1])
    ax1.set_title(f"Buckeyelium Concentration for Final Timestep, t={tstop}s")
    ax1.set_ylabel(r"Concentration ($\frac{g}{m^3}$)")
    ax1.set_xlabel("Distance Along Stream from Source ($m$)")

    # Axes labels and titles to be informative, include legend for seasons
    ax2.set_xlabel('Time ($s$)')
    ax2.set_ylabel('Position ($m$)')
    ax2.set_title("Buckeyelium Concentration Over Time")

    return fig


def get_all_plots():
    '''
    This function runs all solutions to science questions #1, #2, and #3, 
    returning and saving all plots.
    '''

    # Run validation tests, buckeyelium and no advection
    fig_b = buckeyelium(v=0.5, D=0.05, xstop=10., dx=0.1, tstop=40.,
                        dt=0.05)
    fig_b.suptitle("Advection-Diffusion Test")
    fig_b.tight_layout()
    fig_b.savefig("adv-diff_valid.png")

    fig_noadv = buckeyelium(v=0, D=0.05, xstop=10., dx=0.1, tstop=40.,
                            dt=0.05)
    fig_noadv.suptitle("No-Advection Test")
    fig_noadv.tight_layout()
    fig_noadv.savefig("no-adv_valid.png")

    # Run Miller's Creek scenario
    # Miller's creek inputs are default in buckeyelium function,
    # so can just collect from function
    fig_mc = buckeyelium()
    fig_mc.suptitle("Miller's Creek Arsenic Test")
    fig_mc.tight_layout()
    fig_mc.savefig("mill-cre_scen.png")

    # Run filtering scenario with miller's creek inputs
    fig_f = buckeyelium(filter_net=True)
    fig_f.suptitle("Filtered Miller's Creek Arsenic Test")
    fig_f.tight_layout()
    fig_f.savefig("mill-cre_filt.png")