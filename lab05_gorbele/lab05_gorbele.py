#!/usr/bin/env Python3

'''
Lab 05: Snowing Balling
'''

# Starting w/ insolation and varying w/ latitude

# Diffuse heat thru 50m ocean depth

#temp diffuses across latitudes
# heat flux: q = -D(dT/dy)
# Heat diffusion

#dT/dt = lam(d^2T/dy^2)+lam/Axz(dT/dy)(dAxz/dy)
# Last term is spherical coordinate correction

# (__I-delt*lam*__K) == __L
# _Tj+1 = __L^-1(_Tj+delt*lam*_f(_Tj,y))

#delt*lam*_F(_Tj,y) = lam*delt/Axz*1/(4dely^2)(__BTj)(__BAj)
# Evaluated at current time

#dTinsol/dt = 1/(rho*C*dz)*(S(y)(1-alpha)-eps*sig*T^4)
# Change in temperature of surface due to insolation


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

radEarth = 6357000.  # Earth Radius (m)
mxdlyr = 50.         # Depth of mixed layer (m)
sigma = 5.67e-8      # Steffan Boltzmann Constant
C = 4.2e6            # Heat capacity of water (J/m^-3/K)
rho = 1020           # Density of sea water (kg/m^3)
emis = 1             # Emissivity
S0 = 1370            # Solar Flux (W/m^2)
Afrozen = 0.6        # Albedo for Snow/Ice
Awater = 0.3         # Albedo for Water/Ground
lam = 100            # Thermal diffusivity (m^2/s)


# be able to set albedo dynamically (Axz), should be same size as grid
# Based on temp, can say frozen <= 0, A[frozen] = Afrozen; A[~frozen] = Awater


# Bring in new functions
def gen_grid(nbins=18):
    '''
    Generate a grid from 0 to 180 lat (where 0 is south pole, 180 is north)
    where each returned point represents the cell center.

    Parameters
    ----------
    nbins : int, defaults to 18
        Set the number of latitude bins.

    Returns
    -------
    dlat : float
        Grid spacing in degrees.
    lats : Numpy array
        Array of cell center latitudes.
    '''

    dlat = 180 / nbins  # Latitude spacing.
    lats = np.arange(0, 180, dlat) + dlat/2. # Lat cell centers

    # Alternative way to obtain grid:
    # lats = np.linspace(dlat/2., 180-dlat/2, nbins)

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowball_earth(nlat=18, tfinal=10000., dt=1., lam=100., emis=1.0,
                   init_cond=temp_warm, apply_spherecorr=False,
                   apply_insol=False, solar=S0, albuce=0.6, albgnd=.3):
    '''
    
    Parameters
    ----------
    nlat : int, defaults to 18
        Number of latitude cells. Pick something that is 
    tfinal : float, defaults to 10000
        Time length of simulation, in years.
    dt : float, defaults to 1.
        Size of time step, in years.
    lam : float, defaults to 100.
        Ocean diffusivity in m2/s
    emis : float, defaults to 1.0
        Emissivity of Earth.
    init_cond : function, float, or array
        Set the initial condition of the simulation. If a function is given,
        it must take latitudes as input and return temperature as a function
        of lat.
    apply_spherecorr : bool, defaults to False
        Apply spherical correction term.
    apply_insol : bool, defaults to False
        Apply insolation term.
    solar : float, defaults to 1370
        Set level of solar forcing in W/m2
    albice, albgnd : floats, default to 0.6 and 0.3
        Set albedo values for ice and ground

    Returns
    -------
    lats : numpy array
        Latitudes representing cell centers in degrees; 0
    Temp : numpy array
        dsahbdhasd
    '''

    dlat, lats = gen_grid(nlat)
    # Y-spaceing for cells in physical units:
    dy = np.pi * radEarth / nlat

    # Create our first derivative operator.
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0, :] = B[-1, :] = 0

    # Create area array:
    Axz = np.pi * ((radEarth+50.0)**2 - radEarth**2) * np.sin(np.pi/180.*lats)
    # Get derivative of Area:
    dAxz = np.matmul(B, Axz)

    nsteps = int(tfinal / dt)

    # Set timestep to seconds:
    dt = dt * 365 * 24 * 3600

    albedo = np.zeros(nlat)


    # Create temp array, set initial condition
    Temp = np.zeros(nlat)
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp += init_cond


    # Create our X matrix
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    # Boundary conditions:
    K[0, 1], K[-1, -2] = 2, 2
    # Units!
    K *= 1/dy**2

    # Create inverted L matrix
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # SOLVE!
    for istep in range(nsteps):
        # Create spherical coordinates correction term
        if apply_spherecorr:
            sphercorr = (lam*dt) / (4*Axz*dy**2) * np.matmul(B, Temp) * dAxz
        else:
            sphercorr = 0

        if apply_insol:
            radiative = (1-albedo)*solar - emis*sigma*(Temp+273)**4
            Temp += dt * radiative / (rho*C*mxdlyr)

        # Advance solution.
        Temp = np.matmul(Linv, Temp + sphercorr)

    return lats, Temp


def test_functions():
    '''
    Suite of tests
    '''

    print("test gen_grid()")
    print("For npoints=5: ")
    dlat_correct, lats_correct = 36.0, np.array[18., 54., 90., 126., 162]
    results = gen_grid(5)
    if results[0] == dlat_correct:
        print("\tPassed!")
    else:
        print('\tFAILED!')
        print(f"Expected: {dlat_correct}, {lats_correct}")
        print(f"Got: {results}")
    

def problem1():
    '''
    '''
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    lats, temp_diff = snowball_earth()
    lats, temp_diffspcor = snowball_earth(apply_spherecorr=True)
    # Call with constant albedo across surface
    lats, temp_alls = snowball_earth(apply_spherecorr=True, apply_insol=True, 
                                     albice=0.3)

    fig, ax = plt.subplots(1, 1)
    ax.plot(lats-90, temp_init, label='Initial Condition')
    ax.plot(lats-90, temp_diff, label='Diffusion Only')
    ax.plot(lats-90, temp_diffspcor,
            label='Diffusion And Spherical Correction')
    ax.plot(lats-90, temp_alls,
            label='Diffusion, Spherical Correction, and Radiative')
    ax.set_title('Solution after 10000 Years')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel(r'Latitude ($^{\circ}$)')
    ax.legend(loc='best')

