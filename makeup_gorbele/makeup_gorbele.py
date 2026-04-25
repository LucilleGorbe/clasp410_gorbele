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
`$ sim_validation()`, which addresses science question #1.
`$ ### FILL IN HERE`, which addresses science question #2, and
`$ ### FILL IN HERE`, which adresses science question #3.
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
           steps=600, flow_rate=0.3, h0=None, seed=410):
    '''
    This function simulates avalanche dynamics via a discretized analysis of
    the Mohr-Coulomb rule on a 2-dimensional (space and time) array.

    Parameters
    -----
    x : vector of floats
        x location of 
    theta_deg : vector of floats
        Angle of slope for each cell of the mountain's snowpack, in degrees.
    Snf : float
        Global snowfall rate, m/hr
    Cx : float or vector of floats
        Cohesion strength of the snowpack, in Pa.
    linear : boolean
        Linearity or non-linearity of 
    phi_deg : float, defaults to 35.0 degrees

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

    # Store the first time of failure
    t_fail = 9999  # Placeholder value


    # Store total outflow of snow
    outflow = 0.0  # in m


    # Start time loop
    for t in range(steps):
        # Track time
        time = t*dt
        h += Snf

        # Calculate ratio of whole slope
        ratio = get_stability(h, theta_deg=theta_deg, C=Cx, phi_deg=phi_deg)

        # Sweep from the summit to the base of the setup
        for i in range(N-1, -1, -1):
            # Check ratio
            if ratio[i] >= 1.0 or h[i] <= 0.0:
                continue
            carry = h[i] * flow_rate
            h[i] -= carry
            j = i - 1
            while carry > 1e-9 and j >= 0:
                h[j] += carry

                # Re-check receiver stability
                rj = float(get_stability(np.array([h[j]]),
                                         np.array([theta_deg[j]]),
                                         np.array([Cx[j]]), phi_deg))

                # If receiver ratio is now collapsing, spill!
                if rj < 1.0:
                    spill = alpha * flow_rate * h[j]
                else:
                    spill = gamma * h[j]

                # Carry the spill over
                h[j] -= spill
                carry = spill
                j -= 1
            if carry > 0.0 and j < 0:
                outflow += carry

    # Return height array
    return h_t, S_t


def snowplot(npoints=100, linear=True, theta_deg=40.0, phi_deg=35.0, Cx=500.0, Sxpeak=0.1, **kwargs):
    '''
    Creates figure and axes objects with required plots on them and returns
    them to answer each question.

    Parameters
    -----
    npoints : int, defaults to 100
        The number of points along the x-axis for computation.
    linear : boolean, defaults to True
        Decides linear vs curved case for snowpack slope
    Cx : float or vector of floats, defaults to 500.0 Pa
        Cohesion of snow, in units of Pascals
    Sxpeak : float, defaults to 0.1 m/hr
        Snowfall rate across cells, or, if Sxbase is defined, at peak of
        snowpack, in m/hr.
    Sxbase : float, optional
        Snowfall rate at base of snowpack, in m/hr. If not provided, snowfall
        is equal across all cells and defined by Sxpeak.
    '''

    # 1. Create the spatial grid (x in units of m)
    x = np.arange(0, 1000, npoints)

    # Check if base snowfall rate is defined
    Sxbase = kwargs.get('Sxbase')

    # Create snowfall rate depending on definition of base snowfall rate
    if Sxbase is None:
        # Use Sxpeak value as default
        Sx = Sxpeak
    else:
        # Linearly interpolate snowfall rate from peak of mtn to base of mtn
        Sx = np.interp(x, [x.min(), x.max()], [Sxpeak, Sxbase])

    # 2. Define Topography
    # For Question 1: Linear slope at 40 degrees
    # For other questions : Curved slope from 10 to 50 degrees
    if linear:
        theta = np.full_like(x, 40.0)
    else:
        theta = np.interp(x, [0, 1000], [10, 50])

    # Run the simulation and store the outputs
    sim = avasim(x=x, theta_deg=theta, Snf=Sx, Cx=Cx)


def validation():
    '''
    This function validates the avasim solver by comparing modeled results to
    a calculated theoretical critical depth. Answers science question #1.
    '''

    # Define non-default variables
    Cx = 500.0