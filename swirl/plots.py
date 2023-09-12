"""
SWIRL Code
    plots.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains simple plotting routines for the SWIRL Identification class.
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# -----------------------------------------------------

def plot_rortex(swirl, f_quiver=6, save=False):
    """
    Plot the rortex map.

    Parameters
    ----------
    swirl : SWIRL Identification instance
        the SWIRL Identification class instance to plot
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
    nx_grid = swirl.v.x.shape[0]
    ny_grid = swirl.v.x.shape[1]
    xrange = np.arange(0, nx_grid)
    yrange = np.arange(0, ny_grid)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    xgrid = xgrid.T
    ygrid = ygrid.T
    # Access velocity field
    vx = swirl.v.x
    vy = swirl.v.y
    # Create figure
    fig, axes = plt.subplots(ncols=1,
                             nrows=1,
                             figsize=(5, 5),
                             sharex=True,
                             sharey=True
                             )

    # Top plot
    # Quiver plot
    axes.quiver(xgrid[::f_quiver, ::f_quiver],
                ygrid[::f_quiver, ::f_quiver],
                vx[::f_quiver, ::f_quiver],
                vy[::f_quiver, ::f_quiver],
                angles='xy',
                units='xy'
                )
    # Plot Criteria
    vmax = np.max(np.abs(swirl.rortex[0]))*0.7
    im0 = axes.imshow(swirl.rortex[0].T, origin='lower',
                         cmap='PiYG', vmax=vmax, vmin=-vmax)

    # Plot contours
    axes.contour(xgrid,ygrid,swirl.rortex[0],levels=[0], colors=['gray'])
    axes.contour(xgrid,ygrid,-swirl.rortex[0],levels=[0], colors=['gray'])
    # Axes
    axes.set_ylim([0, ny_grid+1])
    axes.set_xlim([0, nx_grid+1])
    #axes.set_aspect('equal')
    axes.tick_params(axis="both", direction="in",
                        which="both", right=True, top=True)
    axes.minorticks_on()
    # Colorbar
    divider = make_axes_locatable(axes)
    cax = divider.new_horizontal(size='3%', pad=0.03)
    fig.add_axes(cax)
    fig.colorbar(im0, cax=cax, orientation='vertical', label=r'Rortex $R$')

    # Figure pads
    fig.tight_layout(h_pad=0.00)
    if save:
        plt.savefig('rortex_map.png', dpi=200,
                    bbox_inches='tight', pad_inches=0.01)
    plt.draw()
    fig.show()
# ------------


def plot_gevc_map(swirl, f_quiver=6, save=False):
    """
    Plot rortex criterion and GEVC map.

    Parameters
    ----------
    swirl : SWIRL Identification instance
        the SWIRL Identification class instance to plot
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
    nx_grid = swirl.v.x.shape[0]
    ny_grid = swirl.v.x.shape[1]
    xrange = np.arange(0, nx_grid)
    yrange = np.arange(0, ny_grid)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    xgrid = xgrid.T
    ygrid = ygrid.T
    # Access velocity field
    vx = swirl.v.x
    vy = swirl.v.y
    # Create figure
    fig, axes = plt.subplots(ncols=1,
                             nrows=1,
                             figsize=(5, 5),
                             sharex=True,
                             sharey=True
                             )

    # Bottom plot
    # grid EVCs
    gevc_x = swirl.gevc_map[0]
    gevc_y = swirl.gevc_map[1]
    gevc_s = swirl.gevc_map[2]
    # Arrange in ascending order
    inds = np.abs(gevc_s).argsort()
    gevc_x = gevc_x[inds]
    gevc_y = gevc_y[inds]
    gevc_s = gevc_s[inds]
    # Coloured scatter plot for EVCs
    vmax = np.max(np.abs(gevc_s))
    im1 = axes.scatter(gevc_x,
                       gevc_y,
                       s=10,
                       c=gevc_s,
                       vmax=vmax,
                       vmin=-vmax,
                       cmap='Spectral_r')
    
    # Quiver plot
    axes.quiver(xgrid[::f_quiver, ::f_quiver],
                ygrid[::f_quiver, ::f_quiver],
                vx[::f_quiver, ::f_quiver],
                vy[::f_quiver, ::f_quiver],
                angles='xy',
                units='xy'
                )
    
    # Axes
    axes.set_ylim([0, ny_grid+1])
    axes.set_xlim([0, nx_grid+1])
    #axes.set_aspect('equal')
    axes.tick_params(axis="both", direction="in",
                     which="both", right=True, top=True)
    axes.minorticks_on()
    
    # Colorbar
    divider = make_axes_locatable(axes)
    cax = divider.new_horizontal(size='3%', pad=0.03)
    fig.add_axes(cax)
    fig.colorbar(im1, cax=cax, orientation='vertical',
                 label=r'Grid cardinality $s$')

    # Figure pads
    fig.tight_layout(h_pad=0.00)
    if save:
        plt.savefig('gevc_map.png', dpi=200,
                    bbox_inches='tight', pad_inches=0.01)
    plt.draw()
    fig.show()
# ------------


def plot_decision(swirl, save=False):
    """
    Plot rho-delta and gamma decision diagrams.

    Parameters
    ----------
    swirl : SWIRL Identification instance
        the SWIRL Identification class instance to plot

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
    delta_opt = swirl.params['cluster_params'][0]
    rho_opt = swirl.params['cluster_params'][1]
    gamma_opt = swirl.params['cluster_params'][2]
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
        plt.savefig('decision_plots.png', dpi=200,
                    bbox_inches='tight', pad_inches=0.01)
    plt.draw()
    fig.show()
# ------------


def plot_vortices(swirl, f_quiver=6, save=False):
    """
    Plot identified vortices.

    Parameters
    ----------
    swirl : SWIRL Identification instance
        the SWIRL Identification class instance to plot
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
    nx_grid = swirl.v.x.shape[0]
    ny_grid = swirl.v.x.shape[1]
    xrange = np.arange(0, nx_grid)
    yrange = np.arange(0, ny_grid)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    xgrid = xgrid.T
    ygrid = ygrid.T
    # Access velocity field
    vx = swirl.v.x
    vy = swirl.v.y
    # Create figure
    fig, ax = plt.subplots(ncols=1, nrows=1, 
                           figsize=(5, 5), 
                           sharex=True,
                           sharey=True)
    # Access_Noise
    noise_x = swirl.noise[2]
    noise_y = swirl.noise[3]
    # Colors
    n_detections = len(swirl)
    color_map = cm.get_cmap('PiYG', n_detections)

    # Main plot
    # Quiver plot
    ax.quiver(xgrid[::f_quiver, ::f_quiver],
              ygrid[::f_quiver, ::f_quiver],
              vx[::f_quiver, ::f_quiver],
              vy[::f_quiver, ::f_quiver],
              angles='xy',
              units='xy'
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
    for vortex in swirl:
        # Scatter Cells
        if vortex.orientation > 0.0:
            vcolor = 0.75
            vedge = 0.9
        else:
            vcolor = 0.25
            vedge = 0.1
        circle = plt.Circle((vortex.center[0],
                             vortex.center[1]),
                             vortex.radius,
                             linewidth=0.3,
                             edgecolor=color_map(vedge),
                             facecolor=color_map(vcolor),
                             alpha=.3
                             )
        # Scatter estimated centers
        ax.scatter(vortex.center[0],
                   vortex.center[1],
                   marker='*',
                   color=color_map(vedge),
                   s=80,
                   edgecolor='k',
                   linewidth=0.1
                   )
        # Add circle
        ax.add_patch(circle)

    # Axes
    ax.set_ylim([0, ny_grid+1])
    ax.set_xlim([0, nx_grid+1])
    #ax.set_aspect('equal')
    ax.tick_params(axis="both", direction="in",
                   which="both", right=True, top=True)
    ax.minorticks_on()
    # Figure pads
    fig.tight_layout(h_pad=0.00)
    # Saving figure
    if save:
        plt.savefig('vortices.png', dpi=200,
                    bbox_inches='tight', pad_inches=0.01, )
    plt.draw()
    fig.show()
# ------------
