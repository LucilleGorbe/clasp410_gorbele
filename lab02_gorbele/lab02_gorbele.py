#!/usr/bin/env python3
'''
This file solves the N-Layer atmosphere problem for Lab 01 and all subparts.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
'''


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]

    return dN1dt, dN2dt

def dNdt_predprey(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra Predator-Prey equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = - c*N[1] + d*N[1]*N[0]

    return dN1dt, dN2dt

def euler_solve(func, N1_init=.5, N2_init=.5, dT=.1, t_final=100.0, **kwargs):
    '''
    <Your good docstring here>

    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : floats, defaults=.5,.5
        The initial population density values of N1 and N2.
    dT : float, default=.1
        Time step in years.
    t_final : float, default=100.0
        Integrate until this final value is reached.

    Returns
    ----------
    time : Numpy array
        Time elapsed in years.
    N1, N2 : Numpy arrays
        Normalized population density solutions.    
    '''

    # Initialize solution vectors.
    time=np.arange(0,t_final, dT)
    N1=np.zeros(time.size)
    N2=np.zeros(time.size)
    N1[0]=N1_init
    N2[0]=N2_init

    # Perform the integration.
    for i in range(1, time.size):
        dN1, dN2 = func(i, [N1[i-1], N2[i-1]], **kwargs)
        N1[i] = N1[i-1] + dN1*dT
        N2[i] = N2[i-1] + dN2*dT

    # Return values to caller.
    return time, N1, N2

def solve_rk8(func, N1_init=.5, N2_init=.5, dT=10, t_final=100.0, \
                a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.

    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
        Largest timestep allowed in years.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values

    Returns
    -------
    time : Numpy array
        Time elapsed in years.
    N1, N2 : Numpy arrays
        Normalized population density solutions.
    '''

    from scipy.integrate import solve_ivp

    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                        args=[a, b, c, d], method='DOP853', max_step=dT)

    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]

    #  Return values to caller.
    return time, N1, N2

def verify_euler_rk8():
    '''
    This function verifies implementation of euler_solve() and solve_rk8()
    and compares the performance of the two models by plotting solutions
    under both the Competition and Predator-Prey population models.
    '''

    # Initialize variables
    N1_0 = 0.3
    N2_0 = 0.6
    a, b, c, d = 1, 2, 1, 3

    timepp, N1pp, N2pp = euler_solve(dNdt_predprey, N1_init=N1_0, 
                                     N2_init=N2_0, dT=0.05, a=a, b=b, c=c, d=d)
    timecmp, N1comp, N2comp = solve_rk8(dNdt_predprey, N1_init=N1_0, 
                                     N2_init=N2_0, dT=0.05, a=a, b=b, c=c, d=d)

    # Plot solutions.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[12,4])

