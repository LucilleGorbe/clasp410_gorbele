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
from matplotlib.colors import ListedColormap
plt.style.use("seaborn-v0_8")

# Generate custom segmented colormap for this project
# can specify colors by names and hex codes
fcolors = ['tan', 'forestgreen', 'crimson']
forest_cmap = ListedColormap(fcolors)

dcolors = ['darkslategrey', 'turquoise', 'charteuse', 'crimson']
disease_cmap = ListedColormap(dcolors)


def spread(nstep=4, isize=3, jsize=3, pspread=1.0, pignite=0., pbare0=0.,
           psurvive=1.):
    '''
    Me when my fire is spreading. Also adaptable for disease modelling spread 
    with different interpretations of the states.

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
    pbare0 : float, defaults to 0.0
        Sets chance that each cell will be barren of forest at t=0.
    psurvive : float, defaults to 1.0
        Sets chance that each cell will become immune/barren, and is the
        effective 'toggle' between forest fire and disease simulation. By 
        lowering this value, toggle death chance 'on'.

    Returns
    -------
    area : 3-D numpy array
        dhaudusahdusahuidhsaiudh <FIX LATER>
    '''

    # Create initial forest/healthy population
    area = np.zeros((nstep, isize, jsize)) + 2

    # Set initial conditions for burning/infected and bare/immune
    # Set bare/immune first:
    bare = pbare0 > rand(isize, jsize)
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

    # Run through time
    for t in range(nstep-1):
        # Check: if there are no active fires/cases, exit loop early
        if (area[t, :, :] == 3).sum() == 0:
            # set all future time steps to current time step
            area[t:] = area[t]
            print(f"No more tiles on fire; simulation terminated early at t={t}")
            break

        # Initialize next time step as current time step
        area[t+1, :, :] = area[t, :, :]
        now = area[t, :, :]

        # Create shifted arrays that allow for better logical indexing
        # Essentially, move all non-boundaries one step closer to boundaries
        # and fill opened slot with 0 to prevent spreading past boundaries
        nsplit = np.vstack((np.zeros(jsize), now[:-1, :]))
        ssplit = np.vstack((now[1:, :], np.zeros(jsize)))
        wsplit = np.hstack((np.zeros(isize).reshape(isize, 1), now[:, :-1]))
        esplit = np.hstack((now[:, 1:], np.zeros(isize).reshape(isize, 1)))

        # Make arrays of all tiles that can spread:
        # Fit chance criteria, Is on fire, and Neighbor is forest/healthy
        can_spread_n = (pspread > rand(isize, jsize)) & (now == 3) & (nsplit == 2)
        can_spread_s = (pspread > rand(isize, jsize)) & (now == 3) & (ssplit == 2)
        can_spread_w = (pspread > rand(isize, jsize)) & (now == 3) & (wsplit == 2)
        can_spread_e = (pspread > rand(isize, jsize)) & (now == 3) & (esplit == 2)

        # grab locations of fires/cases
        sp_n = (can_spread_n).nonzero()
        sp_s = (can_spread_s).nonzero()
        sp_w = (can_spread_w).nonzero()
        sp_e = (can_spread_e).nonzero()

        # Spread (while checking if it can spread)
        area[t+1, sp_n[0]-1, sp_n[1]] = 3
        area[t+1, sp_s[0]+1, sp_s[1]] = 3
        area[t+1, sp_w[0], sp_w[1]-1] = 3
        area[t+1, sp_e[0], sp_e[1]+1] = 3

        # Active fires/cases will become barren/immune (or dead).
        # Uses immune chance rather than death chance because
        # rand() includes 0 but not 1.
        # 1: Barren/Immune, 0: Dead
        area[t+1, now == 3] = (1 * int(psurvive > rand()))

    return area


def firerun():
    '''
    Runs the spread function with disease-scenario calibrations and returns the
    output population array.
    '''

    # Create default step sizes and disease conditions
    nstep, isize, jsize = 150, 300, 300
    pspread, pignite, pbare0 = 0.80, 0.03, 0.03

    # Run the solver with the input dynamics
    forest = spread(nstep=nstep, isize=isize, jsize=jsize, pspread=pspread, 
                    pignite=pignite, pbare0=pbare0)

    return forest


def diseaserun():
    '''
    Runs the spread function with forest-scenario calibrations and returns the
    output forest array.
    '''

    # Create default step sizes and forest conditions
    nstep, isize, jsize = 150, 300, 300
    pspread, ppatient0, pimmune0, psurvive = 0.80, 0.03, 0.03, 0.50

    # Run the solver with the input dynamics
    population = spread(nstep=nstep, isize=isize, jsize=jsize, pspread=pspread,
                        pignite=ppatient0, pbare0=pimmune0, psurvive=psurvive)

    return population


def fprogression(forest):
    '''
    Calculate the time dynamics of a forest fire.

    Parameters
    ----------
    population : 3-D numpy array
        The population under study

    Returns
    -------
    deceased, immune, healthy, infected : numpy vectors of floats
        Percentages of the population that each disease status holds over time.
    '''
    # Get total number of points and make percentage conversion constant
    ksize, isize, jsize = forest.shape
    perconv = 100. / (isize * jsize)

    # Find all spots that have forests (or are healthy people)
    # ...and count them as a function of time.
    loc = forest == 1
    bare = loc.sum(axis=(1, 2)) * perconv

    loc = forest == 2
    forested = loc.sum(axis=(1, 2)) * perconv

    loc = forest == 3
    fire = loc.sum(axis=(1, 2)) * perconv

    return [bare, fire, forested]


def plot_fprogression(forest):
    '''
    Calculate the time dynamics of a forest fire and plot them.
    '''

    # Create figure and axis objects
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Calculate dynamics
    dist = fprogression(forest)

    # Plot these dynamics
    ax.plot(dist[0], label='Bare')
    ax.plot(dist[1], label='On Fire')
    ax.plot(dist[2], label='Forested')

    # Appropriately label plot
    ax.set_xlabel('Time ($Skoogle-Seconds$)')
    ax.set_ylabel('Percent %')
    ax.legend(loc='best')
    ax.set_title('Forest Fire Dynamics of the So Many Woods over Time')

    # Return figure to caller
    return fig


def plot_forest2d(forest_in, itime=0):
    '''
    Given a forest of size (ntime, nx, ny), plot the itime-th moment as a
    2d pcolor plot.

    Parameters
    ----------
    forest_in : numpy array
        The input forest array.
    itime : int, defaults to 0
        The time step of the array to be plotted.

    Returns
    -------
    fig : matplotlib figure object
        The figure that was plotted on.
    '''

    # Create figure and axes
    fig, ax = plt.subplots(1, 1)

    # Add our pcolor plot, save the resulting mappable object.
    map = ax.pcolor(forest_in[itime, :, :], vmin=1, vmax=3, cmap=forest_cmap)

    # Add a colorbar by handing our mappable to the colorbar function.
    cbar = plt.colorbar(map, ax=ax, shrink=.8, fraction=.08,
                        location='bottom', orientation='horizontal')
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Bare/Burnt', 'Forested', 'Burning'])

    # Flip y-axis (corresponding to matrix's x direction and label stuff)
    ax.invert_yaxis()
    ax.set_xlabel('Y Coordinate ($lightyears$) $\\longrightarrow$')
    ax.set_ylabel('X Coordinate ($lightyears$) $\\longrightarrow$')
    ax.set_title(f'The So Many Woods at T={itime:03d}')

    # Return figure object to caller
    return fig


def forest_make_all_2dplots(forest_in, folder='forest/results/'):
    '''
    For every time frame in `forest_in`, create a 2D plot and save the image
    in folder.
    '''

    import os

    # Check to see if folder exists, if not, make it.
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Make each plot by time step
    ntime, nx, ny = forest_in.shape
    for i in range(ntime):
        print(f"\tWorking on plot #{i:04d}")
        fig = plot_forest2d(forest_in, itime=i)
        fig.savefig(f"{folder}/forest_i{i:04d}.png")
        plt.close('all')


def dprogression(population):
    '''
    Calculate the time dynamics of a disease.

    Parameters
    ----------
    population : 3-D numpy array
        The population under study.

    Returns
    -------
    [deceased, immune, healthy, infected] : list of numpy vectors of floats
        Percentages of the population that each disease status holds over time.
    '''

    # Get total number of points and make percentage conversion constant
    ksize, isize, jsize = population.shape
    perconv = 100. / (isize * jsize)

    # Find all spots that have forests (or are healthy people)
    # ...and count them as a function of time.
    loc = population == 0
    deceased = loc.sum(axis=(1, 2)) * perconv

    loc = population == 1
    immune = loc.sum(axis=(1, 2)) * perconv

    loc = population == 2
    healthy = loc.sum(axis=(1, 2)) * perconv

    loc = population == 3
    infected = loc.sum(axis=(1, 2)) * perconv

    return [deceased, immune, healthy, infected]


def plot_dprogression(population):
    '''
    Calculate the time dynamics of a disease and plot them.

    Parameters
    ----------
    population : 3-D numpy array
        The population under study.

    Returns
    -------
    fig : matplotlib figure object
        The figure that was plotted on.
    '''

    # Create figure and axis objects
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Calculate dynamics
    dist = dprogression(population)

    # Plot the population disease dynamics by time.
    ax.plot(dist[0], label='Deceased')
    ax.plot(dist[1], label='Immune')
    ax.plot(dist[2], label='Healthy')
    ax.plot(dist[3], label='Infected')

    # Appropriately label plot
    ax.set_xlabel('Time ($Skoogle-Seconds$)')
    ax.set_ylabel('Percent %')
    ax.legend(loc='best')
    ax.set_title('Population Dynamics of Jeremy\'s Disease Emporium over Time')

    # Return figure to caller
    return fig


def plot_disease2d(population_in, itime=0):
    '''
    Given a population of size (ntime, nx, ny), plot the itime-th moment as a
    2d pcolor plot.

    Parameters
    ----------
    population_in : numpy array
        The population under study.
    itime : int, defaults to 0
        The time step of the array to be plotted.

    Returns
    -------
    fig : matplotlib figure object
        The figure that was plotted on.
    '''

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))

    # Add our pcolor plot, save the resulting mappable object.
    map = ax.pcolor(population_in[itime, :, :], vmin=1, vmax=3, 
                    cmap=disease_cmap)

    # Add a colorbar by handing our mappable to the colorbar function.
    cbar = plt.colorbar(map, ax=ax, shrink=.8, fraction=.08,
                        location='bottom', orientation='horizontal')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Deceased', 'Immune', 'Healthy', 'Infected'])

    # Flip y-axis (corresponding to matrix's x direction and label stuff)
    ax.invert_yaxis()
    ax.set_xlabel('Y Coordinate ($lightyears$) $\\longrightarrow$')
    ax.set_ylabel('X Coordinate ($lightyears$) $\\longrightarrow$')
    ax.set_title(f"Jeremy's Disease Emporium at T={itime:03d}")

    # Return figure object to caller
    return fig


def disease_make_all_2dplots(population_in, folder='disease/results/'):
    '''
    For every time frame in `population_in`, create a 2D plot and save the 
    image in folder.
    '''

    import os

    # Check to see if folder exists, if not, make it.
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Make each plot by time step
    ntime, nx, ny = population_in.shape
    for i in range(ntime):
        print(f"\tWorking on plot #{i:04d}")
        fig = plot_disease2d(population_in, itime=i)
        fig.savefig(f"{folder}/population_i{i:04d}.png")
        plt.close('all')


def compare_pspread_pbare(num=10):
    '''
    Parameters
    ----------
    num : int, defaults to 10
        Sets number of steps to be used in comparison for pspread and pbare.

    Returns
    -------
    fig : matplotlib figure object
        The figure that was plotted on.
    '''
    # Set solver conditions, create fig and axes objects
    nstep, isize, jsize, pignite = 150, 300, 300, 0.03
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Create range of comparisons for pspread and pbare
    prange = np.linspace(0, 1, num=num)

    # Create empty lists for data storage
    bare = []
    forested = []
    enflamed = []

    # Calculate forest makeup at final timestep as function of pspread
    for pspread in prange:
        forest = spread(nstep=nstep, isize=isize, jsize=jsize, pignite=pignite,
                        pspread=pspread, pbare0=0.)
        dist = fprogression(forest)
        # Append final step makeups to vectors for plotting
        bare.append(dist[0][-1])
        forested.append(dist[1][-1])
        enflamed.append(dist[2][-1])

    # Plot forest makeup at final timestep as function of pspread
    ax1.plot(prange, bare, label='Bare', color=fcolors[0])
    ax1.plot(prange, forested, label='Forested', color=fcolors[1])
    ax1.plot(prange, enflamed, label='On Fire', color=fcolors[2])

    ax1.set_ylabel('Percentage (%)')
    ax1.set_xlabel(r'$P_{spread}$')
    ax1.set_title(rf'{nstep}-Step Forest Makeup by $P_spread$')

    # Reset-to-empty for data storage
    bare = []
    forested = []
    enflamed = []

    # Calculate forest makeup at final timestep as function of pbare
    for pbare in prange:
        forest = spread(nstep=nstep, isize=isize, jsize=jsize, pignite=pignite,
                        pspread=0.5, pbare0=pbare)
        dist = fprogression(forest)
        # Append final step makeups to vectors for plotting
        bare.append(dist[0][-1])
        forested.append(dist[1][-1])
        enflamed.append(dist[2][-1])

    # Plot forest makeup at final timestep as function of pbare
    ax2.plot(prange, bare, label='Bare', color=fcolors[0])
    ax2.plot(prange, forested, label='Forested', color=fcolors[1])
    ax2.plot(prange, enflamed, label='On Fire', color=fcolors[2])

    ax2.set_ylabel('Percentage (%)')
    ax2.set_xlabel(r'$P_{bare}$')
    ax2.set_title(rf'{nstep}-Step Forest Makeup by $P_bare$')

    return fig


def compare_psurvive_immune(num=10):
    '''
    Parameters
    ----------
    num : int, defaults to 10
        Sets number of steps to be used in comparison for pspread and pbare.

    Returns
    -------
    fig : matplotlib figure object
        The figure that was plotted on.
    '''
    # Set solver conditions, create fig and axes objects
    nstep, isize, jsize, ppatient0, pspread = 150, 300, 300, 0.03, 0.50
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Create range of comparisons for pspread and pbare
    prange = np.linspace(0, 1, num=num)

    # Create empty lists for data storage
    deceased = []
    immune = []
    healthy = []
    infected = []

    # Calculate population disease stats at final timestep as function of pspread
    for psurvive in prange:
        population = spread(nstep=nstep, isize=isize, jsize=jsize,
                            pignite=ppatient0, pspread=pspread, pbare0=0.,
                            psurvive=psurvive)
        dist = dprogression(population)
        # Append final step makeups to vectors for plotting
        deceased.append(dist[0][-1])
        immune.append(dist[1][-1])
        healthy.append(dist[2][-1])
        infected.append(dist[3][-1])

    # Plot population disease stats at final timestep as function of pspread
    ax1.plot(prange, deceased, label='Deceased', color=dcolors[0])
    ax1.plot(prange, immune, label='Immune', color=dcolors[1])
    ax1.plot(prange, healthy, label='Healthy', color=dcolors[2])
    ax1.plot(prange, infected, label='Infected', color=dcolors[3])

    ax1.set_ylabel('Percentage (%)')
    ax1.set_xlabel(r'$P_{spread}$')
    ax1.set_title(rf'{nstep}-Step Forest Makeup by $P_immune$')

    # Reset-to-empty for data storage
    deceased = []
    immune = []
    healthy = []
    infected = []

    # Calculate population disease stats at final timestep as function of pbare
    for pimmune in prange:
        population = spread(nstep=nstep, isize=isize, jsize=jsize, pignite=ppatient0,
                            pspread=pspread, pbare0=pimmune, psurvive=0.50)
        dist = dprogression(population)
        # Append final step makeups to vectors for plotting
        deceased.append(dist[0][-1])
        immune.append(dist[1][-1])
        healthy.append(dist[2][-1])
        infected.append(dist[3][-1])

    # Plot population disease stats at final timestep as function of pbare
    ax2.plot(prange, deceased, label='Deceased', color=dcolors[0])
    ax2.plot(prange, immune, label='Immune', color=dcolors[1])
    ax2.plot(prange, healthy, label='Healthy', color=dcolors[2])
    ax2.plot(prange, infected, label='Infected', color=dcolors[3])

    ax2.set_ylabel('Percentage (%)')
    ax2.set_xlabel(r'$P_{bare}$')
    ax2.set_title(rf'{nstep}-Step Forest Makeup by $P_immune$')

    return fig
