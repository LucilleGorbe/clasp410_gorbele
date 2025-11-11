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
from numpy.random import rand
plt.style.use("seaborn-v0_8")


def spread(nstep=4, isize=3, jsize=3, pspread=1.0, pignite=0.1, pbare=0):
    '''
    Me when my fire is spreading

    Parameters
    ----------
    nsize : int, defaults to 4
        The temporal extent of the model.
    isize, jsize : ints, defaults to 3 and 3, respectively
        The spatial dimensions of the population under consideration.
    pspread : float, defaults to 1.0
        Sets chance that fire will spread.
    pignite : float, defaults to 0.1
        Sets chance that each cell will be a source of forest fire ignition.
    '''

    # Create our little guys and make them all normalsauce
    area = np.zeros((nstep, isize, jsize)) + 2

    # if nothing is on fire, stop! return only the on-fire steps to reduce load

    # Set initial conditions for burning/infected and bare/immune
    # Set bare/immune first:
    bare = pbare > rand(isize, jsize)
    area[0, bare] = 1

    # Now for burning/infected, include singular ignition point test case
    # Via if statement; 0 chance of ignition indicates single point ignition
    if pignite == 0.:
        # Set initial fire to center:
        area[0, isize//2, jsize//2] = 3
    else:  # Scatter fire randomly:
        ignite = np.zeros((isize, jsize), dtype=bool)
        # Set minimum and maximum number of ignition sites, loop until in range
        while (ignite.sum() == 0) | (ignite.sum() > 0.10 * isize * jsize):
            ignite = pignite > rand(isize, jsize)
        print(f"Starting with {ignite.sum()} points on fire or infected.")
        area[0, ignite] = 3

    perconv = 100. / (isize * jsize)

    perfire = np.zeros(nstep)
    perfire[0] = (area[0, :, :] == 3).sum * perconv
    perforest = np.zeros(nstep)
    perforest[0] = (area[0, :, :] == 2).sum * perconv
    perbare = np.zeros(nstep)
    perbare[0] = (area[0, :, :] == 1).sum * perconv
    perdead = np.zeros(nstep)

    # run thru time....
    for t in range(nstep-1):
        if (area[t, :, :] == 3).sum() == 0:
            print(f"No more tiles on fire; simulation terminated early at t={t}")
            break
        # Initialize next time step as current time step
        area[t+1, :, :] = area[t, :, :]

        # Make arrays of all tiles that can spread
        can_spread_n = pspread > rand(isize, jsize)
        can_spread_s = pspread > rand(isize, jsize)
        can_spread_w = pspread > rand(isize, jsize)
        can_spread_e = pspread > rand(isize, jsize)

        # Set boundaries to be unable to spread beyond boundaries
        can_spread_n[0, :] = False
        can_spread_s[-1, :] = False
        can_spread_w[:, 0] = False
        can_spread_e[:, -1] = False

        # grab locations of fires
        fires = np.transpose((area[t, :, :] == 3).nonzero())
        
        # Spread (while checking if it can spread)
        area[t+1, fires & can_spread_n] = 3

        #for f in fires:
            #f[0]

        #for i in range(isize):
            #for j in range(jsize):

                # is there fire????
                #if area[t, i, j] != 3:
                    # Skip past this lil guy
                    #continue

                # pain and fire spreading time
                # criteria to check: only spread to forest
                # do not exceed bounds
                # Spread north
                #if (pspread > rand()) & (i > 0    ) & (area[t, i-1, j] == 2):
                #    area[t+1, i-1, j] = 3
                ## Spread south
                #if (pspread > rand()) & (i < isize) & (area[t, i+1, j] == 2):
                #    area[t+1, i+1, j] = 3
                ## Spread west
                #if (pspread > rand()) & (j > 0    ) & (area[t, i, j-1] == 2):
                #    area[t+1, i, j-1] = 3
                ## Spread east
                #if (pspread > rand()) & (j < jsize) & (area[t, i, j+1] == 2):
                #    area[t+1, i, j+1] = 3

                #make this better w/ logical indexing, make a random thingymabob
                #pchance > rand & awesome > true !!

                # make that bare!
                #area[t+1, i, j] = 1
        # report percentage of forest on fire, barren, and forested
        


    #tasks:
    #initialize fire randomly or by cell (DONE)
    # set bare (immune) randomly (DONE)
    # track percent of forest burning, bare, and forested (DONE)
    # make cool plot


def plot_progression(forest):
    '''
    report and plot the percentages of things across time
    '''

    ksize, isize, jsize = forest.shape()

    perconv = 100. / (isize * jsize)

    loc = forest == 2
    forested = loc.sum(axis=(1, 2)) * perconv

    loc = forest == 3
    fire = loc.sum(axis=(1, 2)) * perconv

    loc = forest == 1
    bare = loc.sum(axis=(1, 2)) * perconv

    plt.plot(forested, label='Forested')
    plt.plot(fire, label='On Fire')
    plt.plot(bare, label='Bare')
    plt.xlabel('Time ($UNITS$)')  # fill units later
    plt.ylabel('Percent')


def plot_forest2d():
    '''
    Plot the awesome file and stuff
    '''

    pass
