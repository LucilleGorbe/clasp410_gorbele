#!/usr/bin/env python3
'''
This file solves the N-Layer atmosphere problem for Lab 01 and all subparts.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
Ensure that any libraries imported below are installed to the user's
instance of python3.
In a terminal window opened to the directory (folder) that has this script 
and is running python, enter `$ run lab02_gorbele.py`. The $ sign is not typed 
by the user, but indicates where shell input begins.

Plots and results can be obtained by running the commands:
`$ verify_euler_rk8()`, which addresses science question #1. 
`$ vary_comp()`, which addresses science question #2, and
`$ vary_pp()`, which adresses science question #3.
Plots are saved with descriptive names and are labelled.
To show the plots from the terminal, after running one of these commands,
the user can run `$ plt.show()` to display the current plot.

Note that, until the plot is closed or unless interactive mode is turned on, 
the terminal will be occupied running the plot and will not
be able to run any additional functions.
'''


import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
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
    Solves the Lotka-Volterra competition and predator/prey equations using
    the Eulerian method for solving ODEs.

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

    #print("Eulerian solution done")
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

    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                        args=[a, b, c, d], method='DOP853', max_step=dT)

    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]

    # print("rk8 solution done")
    # Return values to caller.
    return time, N1, N2

def verify_euler_rk8():
    '''
    This function verifies implementation of euler_solve() and solve_rk8()
    and compares the performance of the two models by plotting solutions
    under both the Competition and Predator-Prey population models.
    '''

    # Initialize solution initial values.
    N1_0 = 0.3
    N2_0 = 0.6

    # Call solutions. Rely on defaults for a, b, c, d, and t_final args.
    timeppE, N1ppE, N2ppE = euler_solve(dNdt_predprey, N1_init=N1_0, 
                                     N2_init=N2_0, dT=0.05)
    timeppR, N1ppR, N2ppR = solve_rk8(dNdt_predprey, N1_init=N1_0, 
                                     N2_init=N2_0, dT=0.05)
    timecompE, N1compE, N2compE = euler_solve(dNdt_comp, N1_init=N1_0, 
                                     N2_init=N2_0, dT=1.)
    timecompR, N1compR, N2compR = solve_rk8(dNdt_comp, N1_init=N1_0, 
                                     N2_init=N2_0, dT=1.)
    
    # Call solutions with different dTs.
    timeppET, N1ppET, N2ppET = euler_solve(dNdt_predprey, N1_init=N1_0, 
                                     N2_init=N2_0, dT=0.025)
    timeppRT, N1ppRT, N2ppRT = solve_rk8(dNdt_predprey, N1_init=N1_0, 
                                     N2_init=N2_0, dT=0.025)
    timecompET, N1compET, N2compET = euler_solve(dNdt_comp, N1_init=N1_0, 
                                     N2_init=N2_0, dT=0.5)
    timecompRT, N1compRT, N2compRT = solve_rk8(dNdt_comp, N1_init=N1_0, 
                                     N2_init=N2_0, dT=0.5)

    # Plot solutions on two adjacent subplots.
    fig, axes = plt.subplots(2, 2, figsize=[12,8])

    # Plot Lotka-Volterra Model solutions.
    axes[0,0] = plot_LV(axes[0,0], timecompE, timecompR, N1compE, N2compE, \
                        N1compR, N2compR) # Competition
    axes[0,1] = plot_LV(axes[0,1], timeppE,   timeppR,   N1ppE,   N2ppE,   \
                        N1ppR,     N2ppR) # Predator-Prey

    # Provide titles for plots.
    axes[0,0].set_title(r"Lotka-Volterra Competition Model ($dT=1 year$)")
    axes[0,1].set_title(r"Lotka-Volterra Predator-Prey Model ($dT=0.05 years$)")

    # Plot modified dT solutions.     #### INCLUDE dT VALUES AND COEFF VALUES
    axes[1,0] = plot_LV(axes[1,0], timecompET, timecompRT, N1compET, N2compET, \
                        N1compRT, N2compRT) # Competition   
    axes[1,1] = plot_LV(axes[1,1], timeppET,   timeppRT,   N1ppET,   N2ppET,   \
                        N1ppRT,   N2ppRT) # Predator-Prey    

    # Label things well
    axes[1,0].set_title(r"Lotka-Volterra Competition Model ($dT=0.5 years$)")
    axes[1,1].set_title(r"Lotka-Volterra Predator-Prey Model ($dT=0.025 years$)")

    #set coefficients at bottom of plot
    fig.text(0.45, 0.03, r"$a=1, b=2, c=1, d=3$")
    fig.tight_layout()
    plt.savefig('EulerVRK8.png')
    
def plot_LV(ax, timeE, timeR, N1E, N2E, N1R, N2R):
    '''
    This function plots Eulerian and Runge-Kutte models of coupled differential
    equations, though specialized for Lotka-Volterra Models, on an input 
    axis and returns that axis.

    Parameters
    ----------
    ax - plt.ax object
        Axis to be plotted on.
    timeE, timeR - vector of floats
        Time series, in years, for the Eulerian and Runge-Kutte Models.
    N1E, N2E - vectors of floats
        Eulerian solutions for coupled differential equations.
    N1R, N2R - vectors of floats
        Runge-Kutte solutions for coupled differential equations.

    Returns
    -------
    ax - plt.ax object
        Axis to be plotted on.
    '''

    ax.plot(timeE, N2E, 'g'  , label='N2 Euler')
    ax.plot(timeE, N1E, 'b'  , label='N1 Euler')    
    ax.plot(timeR, N1R, 'b--', label='N1 RK8')
    ax.plot(timeR, N2R, 'g--', label='N2 RK8')

    ax.set_xlabel(r"Time ($years$)")
    ax.set_ylabel("Population/Carrying Cap.")
    ax.legend(loc='best')

    # Set axis limits to reasonable values.
    ax.set_ylim(-0.5, 1.5)

    return ax

def vary_comp():
    '''
    This function produces a figure exploration of how the Lotka-Volterra
    Competition Model changes with input coefficients and initial conditions
    using the Runge-Kutta adaptive ODE solver.

    Parameters
    ----------

    Returns
    -------
    fig - matplotlib.pyplot.figure object
        
    '''

    # create the figure
    fig, axes = plt.subplots(4,4,figsize=(12,8))

    # create coefficients and vectors 
    coeffs = [[1,2,1,3],[3, 4, 2, 1],[2,2,2,2],[3,1,3,1]]
    inits = [[0.2, 0.2], [0.8, 0.8], [0.8, 0.2], [0.2, 0.8]]
    dT = 0.05 #maximum dT time step in years 

    # iterate over all coefficients and initial conditions to produce sols
    for i, m in enumerate(coeffs):
        a, b, c, d = m
        for j, n in enumerate (inits):
            N1_0, N2_0 = n
            time, N1, N2 = solve_rk8(dNdt_comp, N1_init=N1_0, N2_init=N2_0,  
                                     dT=dT, t_final=100, a=a, b=b, c=c, d=d)
            ax = axes[i,j]
            ax.plot(time, N1, 'g', label='N1', lw=2)
            ax.plot(time, N2, 'b', label='N2', lw=2)

            ax.set_ylim(0, 1.0)
            ax.legend(loc='best')

        # get rid of excess tick markers that make the graph cluttered
        for axy in axes[:,1:]:
            for l in axy:
                l.yaxis.set_ticklabels([])
        for axx in axes[:-1,:]:
            for k in axx:
                k.xaxis.set_ticklabels([])

        # label coefficients for every group
        fig.text(0.85, (-i)*0.225 + 0.82, f"a = {a},b = {b},\nc = {c},d = {d}", \
                 rotation='horizontal', fontsize='large')

    # label figure informatively
    fig.text(0.46, 0.02, r"Time ($years$)", fontsize='x-large')
    fig.text(0.02, 0.35, "Population/Carrying Cap", rotation='vertical',\
             fontsize='x-large')
    fig.suptitle("  Varying Coefficients and Initial Conditions Produces\
    Different Equilibria in L-V Competition Model", 
                 fontsize='x-large', horizontalalignment='center')
    
    # adjust the figure size and shape for legibility
    fig.tight_layout()
    fig.subplots_adjust(left=0.07, bottom = 0.08)
    
    # save it for the report!
    plt.savefig('vary_comp.png')

def vary_pp():
    '''
    This function produces two figures examining the Lotka-Volterra
    Predator-Prey Model using the Runge-Kutta adaptive ODE solver.
    Solutions are plotted across initial conditions and coefficients, and 
    solutions are plotted against time and each other.
    '''

    # create figures for plotting on
    fig1, axes = plt.subplots(2,2, figsize=(8,8))
    fig2, ax2 = plt.subplots(4, 1, figsize=(4, 8))

    # initialize coefficients and initial values
    coeffs = [[1,2,1,3],[3, 4, 2, 1],[2,2,2,2],[3,1,3,1]]
    n = [[0.2, 0.2], [0.8, 0.8], [0.8, 0.2], [0.2, 0.8]] #initial values
    dT = 0.05 #maximum dT time step in years 

    # iterate over all coefficients fo plotting
    for i, m in enumerate(coeffs):
        a, b, c, d = m
        # create rk8 solutions for plotting
        time1, N11, N21 = solve_rk8(dNdt_predprey, N1_init=n[0][0], N2_init=n[0][1],  
                                     dT=dT, t_final=100, a=a, b=b, c=c, d=d)
        time2, N12, N22 = solve_rk8(dNdt_predprey, N1_init=n[1][0], N2_init=n[1][1],  
                                     dT=dT, t_final=100, a=a, b=b, c=c, d=d)
        time3, N13, N23 = solve_rk8(dNdt_predprey, N1_init=n[2][0], N2_init=n[2][1],  
                                     dT=dT, t_final=100, a=a, b=b, c=c, d=d)
        time4, N14, N24 = solve_rk8(dNdt_predprey, N1_init=n[3][0], N2_init=n[3][1],  
                                     dT=dT, t_final=100, a=a, b=b, c=c, d=d)
        
        #flatten the axis to easily iterate over them and plot all sols
        ax=axes.flatten()[i]
        ax.plot(time1, N11, c='navy',   ls='solid'  , lw=0.8)
        ax.plot(time1, N21, c='navy',   ls='dashed' , lw=0.8) #not very visible
        ax.plot(time2, N12, c='blue',   ls='solid'  , lw=0.8)
        ax.plot(time2, N22, c='blue',   ls='dashdot', lw=0.8)
        ax.plot(time3, N13, c='r',      ls='solid'  , lw=0.8)
        ax.plot(time3, N23, c='r',      ls='dashdot', lw=0.8)
        ax.plot(time4, N14, c='orange', ls='solid'  , lw=0.8)
        ax.plot(time4, N24, c='orange', ls='dashdot', lw=0.8)

        # Change axis limits for visibility; otherwise too occluded
        ax.set_xlim(0,25)

        # Plot legend with generalized style legend and coefficients
        ax.plot([0], [0], 'black' , label='N1')
        ax.plot([0], [0], color='black',ls='--', label='N2')
        ax.legend(loc='upper right')
        ax.set_title(f'a={a}, b={b}, c={c}, d={d}')

        # plot phase diagrams
        ax2[i].plot(N11, N21, c='navy'  )
        ax2[i].plot(N12, N22, c='blue'  )
        ax2[i].plot(N13, N23, c='r'     )
        ax2[i].plot(N14, N24, c='orange')

        # place coefficients next to plots
        fig2.text(0.90, (-i)*0.2 + 0.77, f"a = {a}\nb = {b}\nc = {c}\nd = {d}", \
                 rotation='horizontal', fontsize='large')
        
    # clean up figure and save for later
    fig1.tight_layout()
    fig1.savefig('vary_pp.png')
    fig2.savefig('phase_pp.png')
        