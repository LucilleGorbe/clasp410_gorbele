#!/usr/bin/env Python3

'''
This file doesn't prevent forest fires.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
Ensure that any libraries imported below are installed to the user's
instance of python3.
In a terminal window opened to the directory (folder) that has this script
and is running python, enter `$ run lab03_gorbele.py`. The $ sign is not typed
by the user, but indicates where shell input begins.

Plots and results can be obtained by running the commands:
`$ verify_heatsolve()`, which addresses science question #1.
`$ kanger_diffusion()`, which addresses science question #2, and
`$ kanger_gw_diffusion()`, which adresses science question #3.
Plots are saved with descriptive names and are labelled.
To show the plots from the terminal, after running one of these commands,
the user can run `$ plt.show()` to display the current plot.

Note that, until the plot is closed or unless interactive mode is turned on,
the terminal will be occupied running the plot and will not
be able to run any additional functions.
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
plt.style.use("seaborn-v0_8")

def spread(nstep=4, isize=3, jsize=3, pspread=1.0):
    '''
    Me when my fire is spreading

    Parameters
    ----------
    nsize : int, defaults to 4
        The temporal extent of the model.
    isize, jsize : ints, defaults to 3 and 3, respectively
        The spatial dimensions of the population under consideration.
    pspread : float, defaults to 1.0
        Sets chance that fire will spread
    '''

    #Create our little guys and make them all normalsauce
    area = np.zeros((nstep, isize, jsize)) + 2

    #if nothing is on fire, stop! return only the on-fire steps to reduce load

    # Set initial fire on center [UPDATE FOR LAB LATER]
    area[0, isize//2, jsize//2] = 3
    time = np.arange(nstep)

    for i in range(isize):
        for j in range(jsize):
            # is there fire????
            if area[t, i, j] != 3:
                # Skip past this lil guy
                continue
            #pain and fire spreading time
            #criteria to check: only spread to forest
            # do not exceed bounds
            if pspread > random.rand() & i>0 & area[t, i-1, j] == 2:
                area[t, i-1, j] = 3
            if pspread > random.rand() & i<isize & area[t, i+1, j] == 2:
                area[t, i+1, j] = 3
            if pspread > random.rand() & j>0 & area[t, i, j-1] == 2:
                area[t, i, j-1] = 3
            if pspread > random.rand() & j<jsize & area[t, i, j+1] == 2:
                area[t, i, j+1] = 3

            spreadcheck(area[t:t+1, i-1:i+2, j-1:j+2], i, j, pspread)



def spreadcheck(section, iloc, jloc, pspread):
    '''
    Check if we are spreading babyyyyy
    '''

    # check if guy is forested in this time step
    # check roll for spread (random.rand() function)
        # if ye change to fire
    # check the guy

    return section