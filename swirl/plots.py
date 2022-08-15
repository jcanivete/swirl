"""
SWIRL Code
    plots.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains simple plotting routines for the SWIRL class.
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_decision(swirl, save=False):
    """
    Plot rho-delta and gamma decision diagrams.

    Parameters
    ----------
    swirl : SWIRL instance
        the SWIRL class instance to plot

    save : bool
        set to true to save a .png version of the plot

    Returns
    -------

    Raises
    ------
    """
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(5, 5))

    # Quantities
    rho = swirl.rho
    delta = swirl.delta
    gamma = swirl.gamma
    # Order in gamma order and normalize
    ind = gamma.argsort()
    gamma = gamma[ind]/np.max(gamma)
    rho = rho[ind]/np.max(rho)
    delta = delta[ind]/np.max(delta)
    # Selector params
    rho_opt = swirl.clust_options[1]
    delta_opt = swirl.clust_options[0]
    gamma_opt = swirl.clust_options[2]
    rho_cutoff = np.mean(rho)*rho_opt
    delta_cutoff = np.std(delta)*delta_opt
    gamma_cutoff = 2*rho_cutoff*delta_cutoff
    gamma_cutoff = 2*gamma_opt*np.min(delta)*np.max(rho)

    # Top plot
    # Scatter
    axes[0].scatter(rho, delta, marker='o', color='gray', s=10)
    # Thresholds
    axes[0].axvline(rho_cutoff, linestyle=':', linewidth=1.0, color='orange')
    axes[0].axhline(delta_cutoff, linestyle=':', linewidth=1.0, color='orange')
    xgamma = np.linspace(0.001, 1, 500)
    ygamma = gamma_cutoff/xgamma
    axes[0].plot(xgamma, ygamma, linestyle='--', linewidth=1.0, color='orange')
    # Axes
    axes[0].set_xlim([0, 1.05])
    axes[0].set_ylim([1e-3, 2.0])
    axes[0].set_yscale('log')
    # Labels
    axes[0].set_xlabel(r'$\rho$')
    axes[0].set_ylabel(r'$\delta$')

    # Bottom plot
    # Scatter
    n_gamma = np.arange(gamma.shape[0])
    axes[1].plot(n_gamma, gamma, marker='o', markersize=2.7,
                 color='gray', linestyle='none')
    # Thresholds
    axes[1].axhline(gamma_cutoff, linestyle='--',
                    linewidth=1.0, color='orange')
    # Axes
    axes[1].set_yscale('log')
    axes[1].set_xlim([0, gamma.shape[0]*1.05])
    axes[1].set_ylim([1e-6, 2.0])
    # Labels
    axes[1].set_xlabel(r'$n$')
    axes[1].set_ylabel(r'$\gamma$')

    # Figure pads
    fig.tight_layout(h_pad=0.5)
    # Save figure
    if save:
        plt.savefig('SWIRL_decision.png', dpi=200,
                    bbox_inches='tight', pad_inches=0.01)
    plt.draw()
    fig.show()

# ------------------------------


def plot_evcmap(swirl, f_quiver=6, save=False):
    """
    Plot rortex criterion and GEVC map.

    Parameters
    ----------
    swirl : SWIRL instance
        the SWIRL class instance to plot
    f_quiver : int
        sets the frequency of the quiver plot arrows
    save : bool
        set to true to save a .png version of the plot

    Returns
    -------

    Raises
    ------
    """
    # Create grid
    nx = swirl.v.x.shape[0]
    xrange = np.arange(0, nx+1)
    yrange = np.arange(0, nx+1)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    xgrid[...] = xgrid[...].T
    ygrid[...] = ygrid[...].T
    # Access velocity field
    vx = swirl.v.x
    vy = swirl.v.y
    # Create figure
    fig, axes = plt.subplots(ncols=1,
                             nrows=2,
                             figsize=(5, 8),
                             sharex=True,
                             sharey=True
                             )

    # Top plot
    # Quiver plot
    axes[0].quiver(xgrid[::f_quiver, ::f_quiver],
                   ygrid[::f_quiver, ::f_quiver],
                   vx[::f_quiver, ::f_quiver],
                   vy[::f_quiver, ::f_quiver],
                   angles='xy',
                   units='xy',
                   scale=0.1,
                   width=0.8
                   )
    # Plot Criteria
    vmax = np.max(np.abs(swirl.R[0]))*1.0
    im0 = axes[0].imshow(swirl.R[0].T, origin='lower',
                         cmap='PiYG', vmax=vmax, vmin=-vmax)
    # Axes
    axes[0].set_ylim([0, nx])
    axes[0].set_xlim([0, nx])
    axes[0].set_aspect('equal')
    axes[0].tick_params(axis="both", direction="in",
                        which="both", right=True, top=True)
    axes[0].minorticks_on()
    # Colorbar
    divider = make_axes_locatable(axes[0])
    cax = divider.new_horizontal(size='3%', pad=0.03)
    fig.add_axes(cax)
    fig.colorbar(im0, cax=cax, orientation='vertical', label=r'Rortex $R$')

    # Bottom plot
    # grid EVCs
    xM = swirl.M[0]
    yM = swirl.M[1]
    sM = swirl.M[2]
    # Arrange in ascending order
    inds = np.abs(sM).argsort()
    xM = xM[inds]
    yM = yM[inds]
    sM = sM[inds]
    # Coloured scatter plot for EVCs
    vmax = np.max(np.abs(sM))
    im1 = axes[1].scatter(xM,
                          yM,
                          s=10,
                          c=sM,
                          vmax=vmax,
                          vmin=-vmax,
                          cmap='Spectral_r')
    # Quiver plot
    axes[1].quiver(xgrid[::f_quiver, ::f_quiver],
                   ygrid[::f_quiver, ::f_quiver],
                   vx[::f_quiver, ::f_quiver],
                   vy[::f_quiver, ::f_quiver],
                   angles='xy',
                   units='xy',
                   scale=0.1,
                   width=0.8
                   )
    # Axes
    axes[1].set_ylim([0, nx])
    axes[1].set_xlim([0, nx])
    axes[1].set_aspect('equal')
    axes[1].tick_params(axis="both", direction="in",
                        which="both", right=True, top=True)
    axes[1].minorticks_on()
    # Colorbar
    divider = make_axes_locatable(axes[1])
    cax = divider.new_horizontal(size='3%', pad=0.03)
    fig.add_axes(cax)
    fig.colorbar(im1, cax=cax, orientation='vertical',
                 label=r'Grid cardinality $s$')

    # Figure pads
    fig.tight_layout(h_pad=0.00)
    if save:
        plt.savefig('SWIRL_evcmap.png', dpi=200,
                    bbox_inches='tight', pad_inches=0.01)
    plt.draw()
    fig.show()
##
##
#######################################################
#
#


def plot_vortices(swirl, f_quiver=6, save=False):
    """
    Plot identified vortices.

    Parameters
    ----------
    swirl : SWIRL instance
        the SWIRL class instance to plot
    f_quiver : int
        sets the frequency of the quiver plot arrows
    save : bool
        set to true to save a .png version of the plot

    Returns
    -------

    Raises
    ------
    """
    # Create grid
    nx = swirl.v.x.shape[0]
    xrange = np.arange(0, nx+1)
    yrange = np.arange(0, nx+1)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    xgrid[...] = xgrid[...].T
    ygrid[...] = ygrid[...].T
    # Access velocity field
    vx = swirl.v.x
    vy = swirl.v.y
    # Create figure
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(
        5, 5), sharex=True, sharey=True)
    # Access_Noise
    noise_x = swirl.noise[2]
    noise_y = swirl.noise[3]
    # Colors
    n_detections = len(swirl.vortices)
    color_map = cm.get_cmap('PiYG', n_detections)

    # Main plot
    # Quiver plot
    ax.quiver(xgrid[::f_quiver, ::f_quiver],
              ygrid[::f_quiver, ::f_quiver],
              vx[::f_quiver, ::f_quiver],
              vy[::f_quiver, ::f_quiver],
              angles='xy',
              units='xy',
              scale=0.1,
              width=0.8
              )
    # Scatter Noise
    ax.scatter(noise_x,
               noise_y,
               marker='s',
               s=1,
               edgecolor='k',
               linewidth=1,
               facecolors='Gray',
               alpha=0.2,
               label=r'Noise'
               )
    # Loop over vortices for plotting
    for i in np.arange(n_detections):
        # Scatter Cells
        vcolor = np.mean(swirl.vortices[i].X)/220.+0.5
        if vcolor > 0.5:
            vcolor = 0.75
            vedge = 0.9
        else:
            vcolor = 0.25
            vedge = 0.1
        circle = plt.Circle((swirl.vortices[i].center[0],
                             swirl.vortices[i].center[1]),
                            swirl.vortices[i].r,
                            linewidth=0.3,
                            edgecolor=color_map(vedge),
                            facecolor=color_map(vcolor),
                            alpha=.3,
                            label=r'Vortex '+str(i+1)
                            )
        # Scatter estimated centers
        ax.scatter(swirl.vortices[i].center[0],
                   swirl.vortices[i].center[1],
                   marker='*',
                   color=color_map(vedge),
                   s=80,
                   edgecolor='k',
                   linewidth=0.1
                   )
        # Add circle
        ax.add_patch(circle)

    # Axes
    ax.set_ylim([0, nx])
    ax.set_xlim([0, nx])
    ax.set_aspect('equal')
    ax.tick_params(axis="both", direction="in",
                   which="both", right=True, top=True)
    ax.minorticks_on()
    # Figure pads
    fig.tight_layout(h_pad=0.00)
    # Saving figure
    if save:
        plt.savefig('SWIRL_vortices.png', dpi=200,
                    bbox_inches='tight', pad_inches=0.01, )
    plt.draw()
    fig.show()
