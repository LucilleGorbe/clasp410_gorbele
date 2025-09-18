#!/usr/bin/env python3
'''
This file solves the N-Layer atmosphere problem for Lab 01 and all subparts.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
<tbd>
'''

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Physical Constants
sigma = 5.67E-8 #W/m2/K-4

#function that accepts nlayers, S0, epsilon, and albedo {

def n_layer_atmos(nlayers, epsilon=1, albedo=0.33, s0=1350, debug=False):
    '''
    This function simulates an n-layer atmosphere.
    -----
    Parameters
    -----
    nlayers - int
        The number of atmospheric layers included in this atmosphere model.
    epsilon - float
        Defaults to 1, the emissivity of all layers in this model.
    albedo - float
        Defaults to 0.33, the average reflectivity of the body's surface.
    s0 - int
        Defaults to 1350, the solar forcing constant (W/m2) at the body's top-of-atmosphere.
    debug - bool
        Defaults to false, enters debug mode.
    -----
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

    #Invert matrix
    Ainv = inv(A)
    # get solutions
    fluxes = np.matmul(Ainv, b)

    temp = (fluxes/sigma)**(1/4)
    return temp

temps_4 = n_layer_atmos(4)
#plot temp function

#compare test cases function

#question 3 function

#question 3 function

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
