#!/usr/bin/env python3
'''
This file solves the N-Layer atmosphere problem for Lab 01 and all subparts.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
<tbd>
'''

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

# Physical Constants
sigma = 5.67E-8 #W/m2/K-4

#function that accepts nlayers, S0, epsilon, and albedo {

def n_layer_atmos(nlayers, epsilon=1., albedo=0.33, s0=1350., debug=False):
    '''
    This function simulates an n-layer atmosphere.
    
    Parameters
    -----
    nlayers - int
        The number of atmospheric layers included in this atmosphere model.
    epsilon - float
        Defaults to 1, the emissivity of all layers in this model.
    albedo - float
        Defaults to 0.33, the average reflectivity of the body's surface.
    s0 - float
        Defaults to 1350, the solar forcing constant (W/m2) at the body's top-of-atmosphere.
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
            if (i == j):
                A[i,j] = - 2 + 1 * (i==0) #maybe faster than if statements
            else: 
                A[i,j] = (1-epsilon)**(np.abs(i-j)-1) * epsilon**(i>0)
            # row major: i is tbe row, j is the column, so loops through first row
            #write it so its expressed entirely as I, J, and epsilon
    if debug:
        print(A)
    b[0] = -0.25 * s0 * (1-albedo)
    if debug:
        print(b)

    #Invert matrix
    Ainv = inv(A)
    # get solutions
    fluxes = np.matmul(Ainv, b)
    if debug:
        print(fluxes)

    tempS = (fluxes/(sigma))**(1/4)
    tempS[1:] = tempS[1:] / (epsilon**(1/4)) # Handles non-one emissivity case
    return tempS

#plot temp function
def plot_funct(temps, epsilon=1., albedo=0.33, s0=1350.):
    '''
    This function plots a graph of temperature by layer.

    Parameters
    -----
    temps - vector of floats
        Temperatures by atmospheric layer, starting at layer 0, the surface.
    epsilon - float
        Defaults to 1, the emissivity of all layers in this model.
    albedo - float
        Defaults to 0.33, the average reflectivity of the body's surface.
    s0 - float
        Defaults to 1350, the solar forcing constant (W/m2) at the body's top-of-atmosphere.

    Returns
    -----
    None
    '''
    nlayers = np.size(temps) - 1
    layers = np.arange(0, nlayers+1)

    fig, ax = plt.subplots(1,1,figsize=[6,4])
    ax.plot(temps, layers, label=rf'$\epsilon$={epsilon}, $\alpha$={albedo}, $s0$={s0}')
    ax.set_title(f'Temperature by Layer under {nlayers}-Layer Atmospheric Model.')
    ax.set_ylabel('Layer')
    ax.set_xlabel(r'Temperature ($K$)')
    ax.legend(loc='best')
    fig.tight_layout()


#compare test cases function
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
    Comparison sourced from https://singh.sci.monash.edu/models/Nlayer/N_layer.html
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
def vary_emis_layers(emismin=0., emismax=1., debug=False):
    '''
    
    '''

    #Create test values for later use
    testpilon = 0.255
    testn = 1

    emis=np.arange(emismin, emismax, 0.10)
    layers=np.arange(1,12)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=[12,4])
    tempse=np.zeros(np.size(emis))
    for e in (range(np.size(emis))):
        tempse[e]=n_layer_atmos(testn, epsilon=emis[e])[0] # Append surf. temps only

    if debug:
        print(tempse)
        print(emis)
        
    ax1.plot(emis, tempse, label=rf'$\epsilon$={emis[e]}')
    ax1.set_title()
    ax1.legend(loc='best')

    tempsl=np.zeros(np.size(layers))
    for l in range(np.size(layers)):
        layer = np.arange(0,layers[l]+1) # For plotting
        tempsl = n_layer_atmos(layers[l], epsilon=testpilon)
        ax2.plot(tempsl, layer, label = f'{layers[l]}-Layer System')
    ax2.legend(loc='best')
    ax2.set_title('')
    ax2.set_xlabel('')

    plt.tight_layout()

    

#venus function (q4)

#nuke function (q5)

#__name__ = '__main__'

#lab report
    #intro
    #methodology
        #equations done in LaTeX
    #results
        #"To reproduce my results, run function...""
        #"Now we explore... For one layer..."
        #what does my code teach me
    #Discussion
        #reflects on experiment - different stuff, what is limiting, how compares to other studies
            #how well does this represent the real earth?
            #earth not in energy balance, has shortcomings
            #misses other forms of energy transport
            #winds, weathers, latent heat, terrestrial heat
