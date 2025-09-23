#!/usr/bin/env python3
'''
This file solves the N-Layer atmosphere problem for Lab 01 and all subparts.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
Ensure that any libraries imported below are installed to the user's
instance of python3.
In the terminal, enter `$ run lab01_gorbele.py`. The $ sign is not typed
by the user, but indicates where shell input begins.

Next, to verify the code is working on the user's machine, run 
`$ verify_n_layer()` in the terminal. If the values match, then the code
is ready to run on the user's machine. 

Plots and results can be obtained by running the commands:
`$ vary_emis_layers()`, which addresses science question #1. 
Different values may be input if the user is curious about emissivity ranges.
`$ venus()`, which addresses science question #2, and
`$ nuke()`, which adresses science question #3.
Plots are saved with descriptive names and are labelled.
'''

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

# Physical Constants
sigma = 5.67E-8 #W/m2/K-4

def n_layer_atmos(nlayers, epsilon=1., albedo=0.33, s0=1350., \
                  s0layer=0, debug=False):
    '''
    This function produces a temperature profile 
    based on an n-layer, energy balanced atmosphere.
    
    Parameters
    -----
    nlayers - int
        The number of atmospheric layers included in this atmosphere model.
    epsilon - float
        Defaults to 1, the emissivity of all layers in this model.
    albedo - float
        Defaults to 0.33, the average reflectivity of the body's surface.
    s0 - float
        Defaults to 1350, the solar forcing constant (W/m2) 
        at the body's top-of-atmosphere.
    s0layer - int
        Defaults to 0, the layer of the atmosphere that 
        absorbs solar irradiance in a basic model.
    debug - bool
        Defaults to false, enters debug mode.
    
    Returns
    -----

    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on model
    for i in range(nlayers+1): 
        for j in range(nlayers+1):
            # Handles diagonal
            if (i == j):
                A[i,j] = - 2 + 1 * (i==0)
            # Handles sum functions
            else: 
                A[i,j] = (1-epsilon)**(np.abs(i-j)-1) * epsilon**(i>0)
    if debug:
        print(A)
    b[s0layer] = -0.25 * s0 * (1-albedo)
    if debug:
        print(b)

    #Invert matrix
    Ainv = inv(A)
    # get solutions
    fluxes = np.matmul(Ainv, b)
    if debug:
        print(fluxes)

    tempS = (fluxes/(sigma))**(1/4)
    # Handles perfect absorption of Earth's surface to LW radiation
    tempS[1:] = tempS[1:] / (epsilon**(1/4)) 
    if debug:
        print(tempS)
    return tempS

#plot temp function
def plot_funct(temps, epsilon=1., albedo=0.33, s0=1350.):
    '''
    This function plots a graph of temperature by altitude.

    Parameters
    -----
    temps - vector of floats
        Temperatures by atmospheric layer, starting at layer 0, the surface.
    epsilon - float
        Defaults to 1, the emissivity of all layers in this model.
    albedo - float
        Defaults to 0.33, the average reflectivity of the body's surface.
    s0 - float
        Defaults to 1350, the solar forcing constant (W/m2) 
        at the body's top-of-atmosphere.

    Returns
    -----
    ax - matplotlib.axes._axes.Axes
        Axis being plotted upon. Most modifications handled, 
        returns for later modification.
    '''
    # create layers vector for plotting as a proxy for altitude.
    nlayers = np.size(temps) - 1 
    layers = np.arange(0, nlayers+1)

    # Plot the temperatures as a function of altitude
    fig, ax = plt.subplots(1,1,figsize=[6,4])
    ax.plot(temps, layers, label=rf'$\epsilon$={epsilon} '
            rf'$\alpha$={albedo}, $s0$={s0}')

    # Decorate the plot to make it comprehensible at a glance
    ax.set_title(f'Temperature by Layer under \
                 {nlayers}-Layer Atmospheric Model.')
    ax.set_ylabel('Layer')
    ax.set_xlabel(r'Temperature ($K$)')
    ax.legend(loc='best')
    fig.tight_layout()

    # Return ax for later calls so this function 
    # can be reused in the nuke function
    return ax


def verify_n_layer():
    '''
    This function verifies that n_layer_atmos behaves in line with other
    N-Layer atmospheric simulations through terminal printing.
    
    Parameters
    -----
    None

    
    Returns
    -----
    None

    Notes
    -----
    Comparison sourced from 
    https://singh.sci.monash.edu/models/Nlayer/N_layer.html
    '''
    tempS_est4 = n_layer_atmos(4)[0]
    tempS_ver4 = 375.8 # K
    tempA3_est4 = n_layer_atmos(4)[3]
    tempA3_ver4 = 298.8 # K

    tempS_est6 = n_layer_atmos(6)[0]
    tempS_ver6 = 408.8 # K
    tempA3_est6 = n_layer_atmos(6)[3]
    tempA3_ver6 = 355.4 # K

    tempS_est4_e50 = n_layer_atmos(4, epsilon=0.50)[0]
    tempS_ver4_e50 = 310.6 # K
    tempA3_est4_e50 = n_layer_atmos(4, epsilon=0.50)[3]
    tempA3_ver4_e50 = 251.3 # K

    tempS_est4_A0 = n_layer_atmos(4, albedo=0.0)[0]
    tempS_ver4_A0 = 415.4 # K
    tempA3_est4_A0 = n_layer_atmos(4, albedo=0.0)[3]
    tempA3_ver4_A0 = 330.3 # K

    print('Beginning test cases... \n')

    print('\nVerifying a 4-layer atmosphere simulation...')
    print(f'Surface temperature of 4-layer simulation:                      \
            {tempS_est4:.1f} K') 
    print(f'Surface temperature of 4-layer verification:                    \
            {tempS_ver4} K') 
    print(f'Atmosphere layer 3 temperature of 4-layer simulation:           \
            {tempA3_est4:.1f} K') 
    print(f'Atmosphere layer 3 temperature of 4-layer verification:         \
            {tempA3_ver4} K') 
          
    print('\nVerifying a 6-layer atmosphere simulation...')
    print(f'Surface temperature of 6-layer simulation:                      \
            {tempS_est6:.1f} K') 
    print(f'Surface temperature of 6-layer verification:                    \
            {tempS_ver6} K') 
    print(f'Atmosphere layer 3 temperature of 6-layer simulation:           \
            {tempA3_est6:.1f} K') 
    print(f'Atmosphere layer 3 temperature of 6-layer verification:         \
            {tempA3_ver6} K')  
          
    print('\nVerifying a 0.50 emissivity atmosphere simulation...')
    print(f'Surface temperature of 0.50 emissivity simulation:              \
            {tempS_est4_e50:.1f} K') 
    print(f'Surface temperature of 0.50 emissivity verification:            \
            {tempS_ver4_e50} K') 
    print(f'Atmosphere layer 3 temperature of 0.50 emissivity simulation:   \
            {tempA3_est4_e50:.1f} K') 
    print(f'Atmosphere layer 3 temperature of 0.50 emissivity verification: \
            {tempA3_ver4_e50} K')   
    
    print('\nVerifying a 0.00 albedo atmosphere simulation...')
    print(f'Surface temperature of 0.00 albedo simulation:                  \
            {tempS_est4_A0:.1f} K') 
    print(f'Surface temperature of 0.00 albedo verification:                \
            {tempS_ver4_A0} K') 
    print(f'Atmosphere layer 3 temperature of 0.00 albedo simulation:       \
            {tempA3_est4_A0:.1f} K') 
    print(f'Atmosphere layer 3 temperature of 0.00 albedo verification:     \
            {tempA3_ver4_A0} K') 

#question 3 function
def vary_emis_layers(emismin=0.01, emismax=1., emisnum=100, debug=False):
    '''
    Simulates an energy-balanced 1-layer atmosphere to find best fit for
    Earth's effective emissivity under model.

    Parameters
    -----
    emismin - float
        Defaults to 0.01, minimum emissivity in comparison
    emismax - float
        Defaults to 1, maximum emissivity in comparison.
    emisnum - int
        Defaults to 99, the number of emissivities to plot over.
    debug - bool
        Defaults to false.

    Returns
    -----
    None
    '''

    # Initialize 'block' atmosphere layer, theorized 
    # effective emissivity of the atmosphere, and temperature to match.
    testpilon = 0.255
    testn = 1
    esTemp = 288 # K
    bestEm = 0
    bestEmidx = 0
    bestL = 0
    bestLTemp=0

    # Initialize reasonable emissivity and layers ranges;
    # if the temperature is significantly different, 
    # different ranges may be used.
    emis=np.linspace(emismin, emismax, emisnum)
    layers=np.arange(1,12)

    tempse=np.zeros(np.size(emis))
    for e in (range(np.size(emis))):
        # Append surf. temps only
        tempse[e]=n_layer_atmos(testn, epsilon=emis[e])[0] 
        # Algorithm to determine best fit.
        if (np.abs(tempse[e]-esTemp) < np.abs(tempse[bestEmidx]-esTemp)):
            bestEm=emis[e]
            bestEmidx=e


    if debug:
        print(tempse)
        print(emis)

    # Plot surface temperature by emissivity and report to terminal
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=[12,4])
    ax1.plot(emis, tempse, label=rf'$\epsilon$={emis[e]}')

    # Make plot legible at a glance
    ax1.set_title('Surface Temperature in 1-Layer Increases With Emissivity')
    ax1.set_ylabel(r'Surface Temperature ($K$)')
    ax1.set_xlabel('Emissivity')

    # Plot surface temperature by atmospheric layers and report to terminal
    for l in range(np.size(layers)):
        layer = np.arange(0,layers[l]+1) # For plotting
        tempsl = n_layer_atmos(layers[l], epsilon=testpilon)
        # Algoritm to determine best fit
        if (np.abs(tempsl[0]-esTemp) < np.abs(bestLTemp-esTemp)):
            bestL=l
            bestLTemp = tempsl[0]

        ax2.plot(tempsl, layer, label = f'{layers[l]}-Layer System')
    
    # Make plot legible at a glance
    ax2.legend(loc='best')
    ax2.set_title('Temperature by Layer Increases with Additional Layers;'
                  r'($\epsilon=0.255$)')
    ax2.set_xlabel(r'Temperature ($K$)')
    ax2.set_ylabel('Atmospheric Layer')
    plt.tight_layout()

    plt.savefig('Earth_TbyEmisANDLayer.jpg')

    print("The closest emissivity in a 1-layer atmopshere to Earth's reality " \
          f"is {bestEm:.2f}.")
    print("The closest atmospheric structure for an effective emissivity of " \
          f"0.255 to Earth's Reality is {bestL} layers.")   

    
def venus(debug=False):
    '''
    Simulates Venusian atmosphere with varying numbers of layers, comparing 
    modelled and observed to determine effective atmospheric structure of Venus.

    Parameters
    -----
    debug - bool
        Defaults to false.

    Returns
    -----
    None 

    Notes
    -----
    Venus albedo sourced from A. Mallama, B. Krobusek, H. Pavlov, (2017).
    "Comprehensive wide-band magnitudes and albedos for the planets,
    with applications to exo-planets and Planet Nine", Icarus 282 p19-33.
    '''

    # Venusian atmosphere assumed perfectly absorbing, provide
    # other known values for Venus
    epsV=1
    albV=0.7 
    sV=2600 # W/m2 
    surfTemp = 700 # K

    tempCompare = 0 # initialize comparison tool
    layerCompare = 0 # initialize comparison tool

    # Initialize reasonable layers ranges;
    # if the temperature is significantly different, 
    # different ranges may be used.
    layers=np.arange(2,75,3)

    # Iterate over a range of layers to find best match to 
    # surface temperature of Venus
    for l in range(np.size(layers)):
        tempsl = n_layer_atmos(layers[l], epsilon=epsV, albedo=albV, s0=sV)
        if (np.abs(tempsl[0]-surfTemp) < np.abs(tempCompare-surfTemp)):
            # Save best match to known surface temperature of Venus
            tempCompare = tempsl[0]
            temps_plot = tempsl.copy()
            layerCompare=layers[l]

    if debug: 
        print(f'Temp Closest to surfTemp: {tempCompare}')
        print(f'Temps: {temps_plot}')
        print(f'Layer: {layerCompare}')
    
    # Create layers vector and plot against temps
    layer = np.arange(0,layerCompare+1)
    fig, ax = plt.subplots(1,1,figsize=[6,4])
    ax.plot(temps_plot, layer, label = f'{layerCompare}-Layer System')

    # Make plot legible at a glance    
    ax.set_title('Venusian Temperature Best Approximated by ' \
                 f'{layerCompare}-Layer System')
    ax.set_xlabel(r'Temperature ($K$)')
    ax.set_ylabel('Venus Atmospheric Layer')
    plt.tight_layout()

    plt.savefig('Venus_TbyLayer.jpg')

def nuke():
    '''
    Simulates 5-layer Atmosphere under an energy-balanced nuclear winter 
    scenario - all incoming solar flux is absorbed by the top layer.

    Parameters
    -----
    None

    Returns
    -----
    None
    '''

    # Call n_layer_atmos to generate temperatures with uppermost layer
    # absorbing all solar radiation
    nuked_temps = n_layer_atmos(nlayers=5, epsilon=0.5, s0layer=5)

    # Plot temps by altitude and use returned axis to rename plot title
    ax = plot_funct(nuked_temps, epsilon=0.5)
    ax.set_title(r'5-Layer Atmosphere Model with Total '
                    'Solar Absorption in Top Layer')
    
    plt.savefig('NukedEarth_TbyLayer.jpg')
