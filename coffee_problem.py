'''
Solve the coffee problem to learn how to drink coffee effectively.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def solve_temp(t, Tenv=20., T0=90., k=1/300.):
    '''
    This function returns temperature as a function of time using Newton's Law of Cooling.
    ---------
    Parameters
    ---------
    Tenv - float
        ambient temperature of the air environment
    T0 - float
        starting temperature of the coffee
    k - float
        heat transfer coefficient of the coffee
    t - vector of float
        timeline for which the cooling occurs
    ---------
    Returns
    ---------
    T_coffee - vector of floats
        
    '''
    T_coffee = Tenv + (T0-Tenv)*np.exp(-k*t)

    return T_coffee

def time_to_temp(Tfinal, Tenv=20., T0=90., k=1/300.):
    '''
    This function returns the exact time that the coffee reaches a final, 
    drinkable temperature using Newton's Law of Cooling for maximum sippage.
    ---------
    Parameters
    ---------
    Tfinal - float
        target temperature of coffee
    Tenv - float
        ambient temperature of the air environment
    T0 - float
        starting temperature of the coffee
    k - float
        heat transfer coefficient of the coffee
    ---------
    Returns
    ---------
    t - float
        time in seconds to cool to Tfinal
    '''
    #input the equation all good style
    t = -(1/k)*np.log((Tfinal - Tenv)/(T0 - Tenv))

    return t

def euler_coffee(dt=0.25, T0=90., Tenv=20., k=1/300., tfinal=300):
    '''
    Solve the cooling method using Euler's method

    dt - float
        time steps
    Tenv - float
        ambient temperature of the air environment
    T0 - float
        starting temperature of the coffee
    k - float
        heat transfer coefficient of the coffee
    tfinal - float
        final time for simulating thing
    '''

    time=np.arange(0,tfinal, dt)
    temp=np.zeros(time.size)
    temp[0]=T0

    for i in range(time.size-1):
        temp[i+1]=temp[i]-dt * k*(temp[i]-Tenv) #want to swap this out and hand a function in instead
        #functions can be handed to function as an argument
    return time, temp

def solve_euler(dfx, dt=0.25, t0=0., tfin=300., f0=90., **kwargs):
    '''
    Solve an ordinary diffeq using Euler's method.
    Extra kwargs are passed to the dfx function. 

    Parameters
    ----------
    dfx : function
        A function representing the time derivative of our diffeq. It should 
        take two arguments: the current time and the current value, and 
        return one value: the time derivative at time 't'.
    f0 : float
        Initial condition for our differential equation.
    t0, tfin : float, 0 and 300., respectively
        The start and final times for our solver, respectively.
    dt : float, defaults to 0.25
        Time step in seconds.
        
    Returns
    -------
    t : np.array
        Time in seconds over the entire solution.
    fx : np.array
        Solution as a function of time.
    '''

    time=np.arange(t0,tfin, dt)
    fx=np.zeros(time.size)
    fx[0]=f0

    for i in range(time.size-1):
        fx[i+1]=fx[i] + dt * dfx(time[i], fx[i], **kwargs) #k*(temp[i]-Tenv) 
        #want to swap this out and hand a function in instead
        #functions can be handed to function as an argument
    return time, fx

def solve_rk8(dfx, dt=0.25, t0=0., tfin=300., f0=90., **kwargs):
    '''
    Solve an ordinary diffeq using Euler's method.
    Extra kwargs are passed to the dfx function. 

    Parameters
    ----------
    dfx : function
        A function representing the time derivative of our diffeq. It should 
        take two arguments: the current time and the current value, and 
        return one value: the time derivative at time 't'.
    f0 : float
        Initial condition for our differential equation.
    t0, tfin : float, 0 and 300., respectively
        The start and final times for our solver, respectively.
    dt : float, defaults to 0.25
        Time step in seconds.
        
    Returns
    -------
    t : np.array
        Time in seconds over the entire solution.
    fx : np.array
        Solution as a function of time.
    '''

    from scipy.integrate import solve_ivp


    result = solve_ivp(dfx, [t0, tfin], [f0], method='DOP853', max_step=dt)


    return result.t, result.y[0, :] #syntax for solving a sys of diffeqs

def newtcool(t,T_now, T_env=20.,k=1/300.):
    '''
    Newton's law of cooling: given time t, Temperature now (Tnow), a cooling
    coefficient (k), and an envirommental temp (T_env), return the rate of
    cooling (i.e., dT/dt)
    '''

    return -k * (T_now - T_env)

def verify_code():
    '''
    Verify that the code is implemented correct
    '''

    t_real = 60. * 10.76
    k = np.log(95./110.) / -120.
    t_code = time_to_temp(120., T0=180., Tenv = 70., k=k)

    print(f"Target solution is: {t_real}. \nNumerical solution is: {t_code}. \
          \nDifference is: {t_real-t_code}")

def explore_numerical_solve(dt=1.):
    '''
    Test different coffee problem scenarios

    Parameters
    -----
    dt : float, defaults to 1.
        Set the timestep for the Euler solver
    '''
    #test different scenarios
    #do it quantitatively to sceen first:
    t1 = time_to_temp(65)           # add cream at T=65 to get to 60
    t2 = time_to_temp(60, T0=85)    # add cream right away
    t3 = time_to_temp(60)           # control case, no cream

    etime, etemp=solve_euler(newtcool,tfin=300.,dt=dt)
    rtime, rtemp=solve_rk8(newtcool,tfin=300.,dt=dt, T_env=0.)

    print(f"Times to drinkable coffee:\n\t \
            Cream added at T=65: {t1:.2f}s\n\t \
            Cream added immediately: {t2:.2f}s\n\t \
            No cream added: {t3:.2f}s")

    #Begin solving problem
    t = np.arange(0, 300., 0.5)
    temp1 = solve_temp(t)           # also same as control case

    fig, ax = plt.subplots(1,1, figsize=[10.24, 5.91])
    ax.plot(t, temp1, label=f'Analytical Solution')
    ax.plot(etime, etemp, '--',  label=rf'Euler solution for $\Delta t={dt}s$')
    ax.plot(rtime, rtemp, '--',  label=rf'RK8 Solution')
    #holy SHIT that took us two steps to get to the solution

    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (C)')
    ax.set_title('Analytical vs Numerical')
    #print(tfin)
    return fig