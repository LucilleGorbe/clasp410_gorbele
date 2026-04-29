'''
This file builds and validates a 1-D avalanche model and explores the impact
of snow cohesion and precipitation on avalanches.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
Ensure that any libraries imported below are installed to the user's
instance of python3.
In a terminal window opened to the directory (folder) that has this script
and is running python, enter `$ run makeup_gorbele.py`. The $ sign is not typed
by the user, but indicates where shell input begins.

Plots and results can be obtained by running the commands:
`$ validation()`, which addresses science question #1.
`$ curved()`, which addresses science question #2, and
`$ runout()`, which adresses science question #3.
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

# Define constants
g = 9.81  # m*s^-2, gravity
rho_snow = 300  # kg*m^-3, typical density of snow from Lab Manual


# Stability analysis function
def get_stability(h, theta_deg, C, phi_deg=35):
    '''
    This function analyzes the stability of a slope of snow using the 
    Mohr-Coulumb Rule.

    Parameters
    -----
    h : vector of floats
        Heights of each cell of the mountain's snowpack, in m. Is only one
        temporal step of the full solution.
    theta_deg : float or vector of floats
        Angle of slope for each cell of the mountain's snowpack, in degrees.
    C : float or vector of floats
        Cohesion strength of the snowpack, in Pa.
    phi_deg : float, defaults to 35
        Friction angle of snowpack, in degrees.

    Returns
    -----
    S : vector of floats
        Stability ratio of available shear strength to shear stress for each
        cell of the mountain's snowpack, unitless.
    '''
    # Convert angles in degrees into radians
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)

    # Calculate the shear stress term
    # Shear stress = component in same plane as cross section
    # kg/m^3 * m/s^2 * m = Pa
    stress = rho_snow * g * h * np.sin(theta_rad)  # Pascals

    # Normal stress = component normal to plane of cross section
    normal = rho_snow * g * h * np.cos(theta_rad)  # Pascals

    # Available shear strength = cohesion + normal force * friction factor
    strength = C + normal * np.tan(phi_rad)  # Pascals

    # Calculate and return stability ratio
    S = strength / stress
    return S


def avasim(x, theta_deg, Snf, Cx, phi_deg=35.0, dt=0.1,
           steps=600, flow_rate=0.3, h0=None):
    '''
    This function simulates avalanche dynamics via a discretized analysis of
    the Mohr-Coulomb rule on a 2-dimensional (space and time) array.

    Parameters
    -----
    x : vector of floats
        x location of the slope, in meters. Spatial location is arbitrary.
    theta_deg : vector of floats
        Angle of slope for each cell of the mountain's snowpack, in degrees.
    Snf : vector of floats
        Snowfall rate, m/hr
    Cx : vector of floats
        Cohesion strength of the snowpack, in Pascals.
    phi_deg : float, defaults to 35.0 degrees
        Friction angle of snowpack.
    dt : float, defaults to 0.1 hrs
        Time step of the solver in hours
    steps : int, defaults, to 600
        Amount of time steps taken by the solver
    flow_rate : float, defaults to 0.3 1/hr
        Ratio of mass that transfers down cells when spilling
    '''

    # Capture all x array points to iterate thru
    N = x.size

    # Initialize slope vector
    h = np.zeros(N)
    if h0 is not None:
        # Add h0
        h += h0

    # Store history of snowfall accumulation and stability as well as time vec.
    h_t = np.zeros((steps, N))
    S_t = np.zeros((steps, N))
    time = np.linspace(0, dt*steps, steps)

    # Store initial stability and heights
    h_t[0] = h
    S_t[0] = get_stability(h, theta_deg=theta_deg, C=Cx, phi_deg=phi_deg)

    # Init first fail time and location storage
    fail_time = None  # Take a page out of Rouhan's book with the use of None
    fail_x = None
    fail_time_idx = None

    # Store total outflow of snow
    outflow = 0.0  # in m

    # Define gamma and alpha,
    # parameters of how much snow gets carried down past initial cell of spill.
    # These parameters are arbitrary, and could be replaced with parameters
    # with stronger physical bases in other iterations of the solver.
    alpha = 0.45
    gamma = 0.10

    # Start time loop from t=1
    for t in range(1, steps):
        # Add snowfall at the top of each time loop
        h += Snf

        # Calculate ratio of whole slope
        ratio = get_stability(h, theta_deg=theta_deg, C=Cx, phi_deg=phi_deg)

        # Sweep from the summit to the base of the setup
        # Assumes time of collapse and movement is much faster than snowfall
        for i in range(N-1, -1, -1):
            # Check ratio, if stable then go past this one
            if ratio[i] >= 1.0 or h[i] <= 0.0:
                continue
            
            # If ratio is now failing, make None failure time and point into
            # the reported failure time and location.
            if fail_time is None:
                fail_time = t*dt  # hrs
                fail_x = i  # meters
                fail_time_idx = t  # index for computing L later
            # Unstable cells send snow downward in a failure
            carry = h[i] * flow_rate
            h[i] -= carry
            j = i - 1
            while carry > 1e-9 and j >= 0:
                # Push significant carry into receiver cell
                h[j] += carry

                # Re-check receiver stability
                rj = float(get_stability(np.array([h[j]]),
                                         np.array([theta_deg[j]]),
                                         np.array([Cx[j]]), phi_deg))

                # If receiver ratio is now unstable, continue spill
                if rj < 1.0:
                    spill = alpha * flow_rate * h[j]
                else:
                    # If not now unstable, push weaker spill
                    spill = gamma * flow_rate * h[j]

                # Carry the spill over
                h[j] -= spill
                carry = spill
                j -= 1
            if carry > 0.0 and j < 0:
                outflow += carry

        # Store height and ratio
        h_t[t] = h
        S_t[t] = ratio

    # Return height, stability, and time arrays, along with total outflow
    # And failure timing + location along slope
    # Transpose h_t and S_t such that x is on the x-axis
    return h_t.transpose(), S_t.transpose(), time, outflow, (fail_time, fail_x, fail_time_idx)


def snowplot(npoints=100, theta_deg=40.0, curved=False, phi_deg=35.0, Cxpeak=500.0, Sxpeak=0.1, 
             dt=0.1, steps=600, flow_rate=0.3, h0=None, **kwargs):
    '''
    Creates figure and axes objects with required plots on them and returns
    them to answer each question.

    Parameters
    -----
    npoints : int, defaults to 100
        The number of points along the x-axis for computation.
    curved : boolean, defaults to False
        Decides constant vs curved case for snowpack slope
    phi_deg : float, defaults to 35.0 degrees
        Friction angle of snowpack.
    Cxpeak : float or vector of floats, defaults to 500.0 Pa
        Cohesion of snow across cells, or, if Cxbase is defined, at peak of
        snowpack, in Pascals
    Sxpeak : float, defaults to 0.1 m/hr
        Snowfall rate across cells, or, if Sxbase is defined, at peak of
        snowpack, in m/hr.
    dt : float, defaults to 0.1 hrs
        Time step of the solver in hours
    steps : int, defaults, to 600
        Amount of time steps taken by the solver
    flow_rate : float, defaults to 0.3 1/hr
        Ratio of mass that transfers down cells when spilling
    h0 : float or vector of floats, defaults to 0m
        Starting height of the snowpack
    Cxbase : float, optional
        Cohesion of snow at bottom of snowpack, in Pascals. If not provided,
        cohesion is equal across all cells and defined by Cxpeak.  
    Sxbase : float, optional
        Snowfall rate at bottom of snowpack, in m/hr. If not provided, snowfall
        is equal across all cells and defined by Sxpeak.

    Returns
    -----
    fig : matplotlib figure object
        The figure that is plotted on
    time : vector of floats
        Timeline for the occurrence
    '''

    # 1. Create the spatial grid (x in units of m)
    x = np.linspace(0, 1000, npoints)

    # Check if base snowfall rate is defined
    Sxbase = kwargs.get('Sxbase')
    Cxbase = kwargs.get('Cxpeak')

    # 2. Define Topography
    # For Question 1: Linear slope at 40 degrees
    # For other questions : Curved slope from 10 to 50 degrees
    if not curved:
        theta = np.full_like(x, theta_deg)
    else:
        theta = np.interp(x, [0, 1000], [10, 50])

    # Create snowfall rate depending on definition of base snowfall rate
    if Sxbase is None:
        # Use Sxpeak value as constant snowfall
        Sx = np.full_like(x, Sxpeak)
    else:
        # Linearly interpolate snowfall rate from base of mtn to peak of mtn
        Sx = np.interp(x, [0, 1000], [Sxbase, Sxpeak])

    # Create snow cohesion depending on definition of base cohesion
    if Cxbase is None:
        # Use Sxpeak value as constant cohesion
        Cx = np.full_like(x, Cxpeak)
    else:
        # Linearly interpolate snowfall rate from base of mtn to peak of mtn
        Cx = np.interp(x, [0, 1000], [Cxbase, Cxpeak])

    # Run the simulation and store the outputs
    h_t, S_t, time, outflow, fail = avasim(x=x, theta_deg=theta, Snf=Sx, Cx=Cx,
                                           phi_deg=phi_deg, dt=dt, steps=steps,
                                           flow_rate=flow_rate, h0=h0)

    # Make four plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot h_t over time and space
    h_t_pcolor = axes[0, 0].pcolor(time, x, h_t, cmap='bone',
                                   shading='auto')
    h_cbar = fig.colorbar(h_t_pcolor, ax=axes[0, 0])
    h_cbar.set_label("Height (m)")
    axes[0, 0].set_title('Height of Snowpack over Time')
    axes[0, 0].set_ylabel("Location from Base (m)")
    axes[0, 0].set_xlabel("Time (hrs)")

    # Plot max h_t over time, along with h_crit
    axes[0, 1].plot(time, h_t.max(axis=0), label='Max. Height')
    axes[0, 1].set_title("Maximum Height of Snowpack over Time")
    axes[0, 1].set_xlabel("Time (hrs)")
    axes[0, 1].set_ylabel("Height of Snowpack (m)")

    # Plot S_t over time and space
    S_t_pcolor = axes[1, 0].pcolor(time, x, S_t, cmap='summer_r', 
                                   shading='auto')
    S_cbar = fig.colorbar(S_t_pcolor, ax=axes[1, 0])
    S_cbar.set_label("Stability")
    axes[1, 0].set_title('Stability of Snowpack over Time')
    axes[1, 0].set_ylabel("Location from Base (m)")
    axes[1, 0].set_xlabel("Time (hrs)")

    # Plot min S_t over time, along with S_crit
    axes[1, 1].plot(time, S_t.min(axis=0), c='yellow', label="Stability")
    axes[1, 1].plot(time, np.full_like(time, 1.0,), c='k', ls='--', 
                    label="Critical Stability")
    axes[1, 1].set_title("Minimum Stability of Snowpack over Time")
    axes[1, 1].set_xlabel("Time (hrs)")
    axes[1, 1].set_ylabel("Stability of Snowpack")
    axes[1, 1].legend(loc='best')

    fig.tight_layout()

    # Report outflow, first fail time and location, and furthest location of
    print("Total outflow:", outflow, "m of snow.")
    if fail[0] is not None:
        print("First failure time:", fail[0], "Hours")
        print("First failure location:", fail[1], "m from base of slope")
        # For Q3: L, runout dist, is defined as first failure location to final
        # nonzero accumulation. "Nonzero" is defined as 0.03m for our purposes.
        print("Runout distance:", fail[1] - x[h_t[:,fail[2]] > 0.03][0], "m")

    return fig, time
    

def validation(npoints=100):
    '''
    This function validates the avasim solver by comparing modeled results to
    a calculated theoretical critical depth. Answers science question #1.

    Parameters
    -----
    npoints : int, defaults to 100
        The number of points along the x-axis for computation.

    Returns
    -----
    fig : matplotlib figure object
        The figure that is plotted on
    '''

    # Calculate h_crit as outlined in science question #1
    C = 500  # Pascals
    theta = 40  # Degrees
    phi = 35  # Degrees

    # Calculate and report h_crit
    h_crit = C / (rho_snow * g * (np.sin(np.radians(theta)) - np.cos(np.radians(theta)) * np.tan(np.radians(phi))))
    print("h_crit =", h_crit, "m")

    # Run simulation and capture figure and axes
    # Using no inputs, as the defaults are fine this time
    fig, time = snowplot(npoints=npoints)
    fig.axes[1].plot(time, np.full_like(time, h_crit), c='r', ls='--', label=f"h_crit={h_crit} m")
    fig.axes[1].legend(loc='best')

    return fig

def curved():
    '''
    This function modifies the model to use spatially varying slope. It then
    tests snow properties, snowfall distribution and cohesion, to see how
    their variation might impact avalanche dynamics. Answers question #2.
    '''