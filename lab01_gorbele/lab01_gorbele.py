#!/usr/bin/env pyton3
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

def n_layer_atmos(nlayers, epsilon=1, albedo=0.33, s0=1350):
    '''
    
    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on model
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            #A[i,j] = 

    #b = 

#}

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


    