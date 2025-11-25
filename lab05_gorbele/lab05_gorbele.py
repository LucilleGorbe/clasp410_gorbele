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

    # Set initial temp curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    # Get base grid:
    npoints = T_warm.size
    dlat, lats = gen_grid(npoints)

    coeffs = np.polyfit(lats, T_warm, 2)

    # Return fit:
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
                   apply_insol=False, solar=S0, albice=0.6, albgnd=.3):
    '''
    This function does things and makes it cold. Brr!

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
        Set level of TOA insolation in W/m2
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

    # Set number of time steps:
    nsteps = int(tfinal / dt)

    # Set timestep to seconds:
    dt = dt * 365 * 24 * 3600

    # Create insolation:
    insol = insolation(solar, lats)

    # Create temp array, set initial condition
    Temp = np.zeros(nlat)
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp += init_cond

    # Set initial albedo.
    albedo = np.zeros(nlat)
    loc_ice = Temp <= -10  # Sea water freezes at ten below.
    albedo[loc_ice] = albice
    albedo[~loc_ice] = albgnd

    # Create our K matrix:
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
            radiative = (1-albedo)*insol - emis*sigma*(Temp+273)**4
            Temp += dt * radiative / (rho*C*mxdlyr)

        # Advance solution.
        Temp = np.matmul(Linv, Temp + sphercorr)

    return lats, Temp


def test_functions():
    '''
    Suite of tests
    '''

    print("test gen_grid():")
    print("For npoints=5: ")
    dlat_correct, lats_correct = 36.0, np.array([18., 54., 90., 126., 162])
    results = gen_grid(5)
    if results[0] == dlat_correct:
        print("\tPassed!")
    else:
        print('\tFAILED!')
        print(f"Expected: {dlat_correct}, {lats_correct}")
        print(f"Got: {results}")



    # TO DO: make a test case for each function up there to make sure it works 
    print("\ntest temp_warm():")
    print("njsnjdnjas")


def problem1():
    '''
    Answers problem 1! does some plotting la la la :D

    "Code up the solver shown in Equation 4. Do this in parts: begin with only
    the basic diffusion solver (Equation 2). Use the values given in Table 1
    and try to reproduce the red line in Figure 1. For this part, use an
    albedo of 0.3 at all points on your grid. Then, add in the spherical
    correction term (Equation 3) and work until you can reproduce the gold line
    in Figure 1. Finally, include the radiative forcing term (Equation 4);
    work to reproduce the green line in Figure 1. Include your validation
    steps in your lab report."

    Returns
    -------
    fig : matplotlib figure object
        The figure to be plotted on.
    '''

    # Create a baseline warm earth without diffusion
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    # Create diffusion only and plus-spherical-correction modes
    lats, temp_diff = snowball_earth()
    lats, temp_diffspcor = snowball_earth(apply_spherecorr=True)
    # Call plus-solar-forcing with constant albedo across surface
    lats, temp_alls = snowball_earth(apply_spherecorr=True, apply_insol=True,
                                     albice=0.3)

    # Create plot and plot all lines for different modes of earth solver
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats-90, temp_init, label='Initial Condition')
    ax.plot(lats-90, temp_diff, label='Diffusion Only')
    ax.plot(lats-90, temp_diffspcor,
            label='Diffusion And Spherical Correction')
    ax.plot(lats-90, temp_alls,
            label='Diffusion, Spherical Correction, and Radiative')
    
    # Label the plot appropriately
    ax.set_title('Solution after 10000 Years')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel(r'Latitude ($^{\circ}$)')
    ax.legend(loc='best')

    # Return figure to caller
    return fig


def problem2():
    '''
    bWah Bwah bwah bwah vary vary vary

    Tune your model so that it can reproduce the warm-Earth equilibrium. There
    are two parameters in our model that contain much uncertainty: diffusivity
    (λ) and emissivity (ϵ). Explore each independently (allow λ to range from
    0 to 150 and ϵ to range from 0 to 1) to determine their impact on the
    equilibrium solution. Then, pick a value for each that, when used in
    combination, allows you to best reproduce the ”warm Earth” curve given by
    temp_warm(). Report your findings and use these values for rest of the lab.

    Returns
    -------
    fig : matplotlib figure object
        The figure to be plotted on.
    '''

    lconst = 25
    econst = 0.5

    nlats = 18

    # Create warm earth baseline
    dlats, lats = gen_grid(nbins=nlats)
    warm_baseline = temp_warm(lats)

    # vary lambda
    lam = np.linspace(0, 150, 75)
    emis = np.linspace(0, 1, 25)

    # So we run the snowball earth model with everything ON and then take their
    # average absolute difference from temp_warm (RSE) and then plot it against
    # lambda and emissivity

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Create empty arrays to store residuals
    lamResids = np.zeros(lam.size)
    emisResids = np.zeros(emis.size)

    # Vary lambda, keep emis & albedo constant
    for pos, l in enumerate(lam):
        # Grab temperatures
        lats, temp_vlam = snowball_earth(nlat=nlats, lam=l, emis=econst,
                                         apply_spherecorr=True,
                                         apply_insol=True, albice=0.3)
        lamResids[pos] = np.sum(np.abs(temp_vlam - warm_baseline))

    # Vary emissivity, keep lambda & albedo constant
    for pos, e in enumerate(emis):
        lats, temp_vemis = snowball_earth(nlat=nlats, lam=lconst, emis=e,
                                          apply_spherecorr=True,
                                          apply_insol=True, albice=0.3)
        emisResids[pos] = np.sum(np.abs(temp_vemis - warm_baseline))

    emisAvgResids = emisResids / nlats
    lamAvgResids = lamResids / nlats

    ax1.plot(lam, lamAvgResids)
    ax1.set_xlabel(r'${\lambda}$ ($\frac{m^2}{s})')
    ax1.set_ylabel('Residuals from temp_warm()')
    ax1.set_title('')

    ax2.plot(emis, emisAvgResids)
    ax2.set_xlabel('Emissivity')
    ax2.set_ylabel('Residuals from temp_warm()')
    ax2.set_title('')

    # Find optimal lambda and emissivity and avg residual for those
    lamopt = lam[np.argmin(lamAvgResids)]
    emisopt = emis[np.argmin(emisAvgResids)]
    lats, temp_opt = snowball_earth(nlat=nlats, lam=lamopt, emis=emisopt,
                                    apply_spherecorr=True,
                                    apply_insol=True, albice=0.3)
    
    optAvgResid = np.sum(np.abs(temp_opt - warm_baseline)) / nlats

    # TO DO: add check to see if this is better than optimal w/ baseline lambda
    # And baseline emissivity
    # If not, use the baseline lambda and optimal emis
    # or the baseline emis and opt. lambda (whichever is better) 

    print(f"Optimal lambda: {lamopt} m^2/s\nOptimal emissivity: {emisopt}")
    print(f"Average Residual for optimal choices: {optAvgResid} Degrees C")

    return fig

        



    