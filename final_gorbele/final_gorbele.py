'''
Mi final project :D
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


def solve_heat(func_0, func_lb, xstop=100., tstop=150*365., dx=1,
               dt=0.1, c2=(0.25*10**-6)*86400, v=0.02, ifconc=0.3, **kwargs):
    '''
    Solves the 1-dimensional diffusion equation.
    New dangerous forever chemical called Buckeyeallium
    Crimson (sometimes red color), smells bad, horribly processed
    Used in making OSU brand sports drinks

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
    v : float, default = 0.02 m/s
        velocity :D
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
    # Will potentially need to change due to intro of 1st order error
    dtmaxdiff = dx**2 / (2*c2)
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
    # print(U[:, 0])

    # THIS CHANGES. NEED TO HAVE A DT TO THIS PER TIME STEP.
    # Set upper boundary conditions across time
    #U[0, :] = func_ub(t, **kwargs)
    U[-1, :] = ifconc  # LOWER BOUNDARY


    # Get r and p coeffs:
    r = c2 * (dt/dx**2)
    p = v * (dt/dx)

    # Solve heat equation
    for j in range(N-1):
        #U[M-1, j+1] = 0  # Maybe??? We lose stuff here but we can never gain it
        U[1:M-1, j+1] = (1-2*r-p) * U[1:M-1, j] + r*U[2:M, j] + (r+p)*U[:M-2, j]
        # Use Neumann conditions as applicable (NOT APPLICABLE)
        # APPLICABLE FOR TOP FOR OUTFLOW.
        #if n:
            #U[0, j+1] = U[1, j+1]
        U[M-1, j+1] = U[M-2, j+1]

    # Return time and position vectors, and temperature array
    return t, x, U