#######################################################
###
###             plots.py
###
#######################################################
##
##  J. R. Canivete Cuissa
##  10.02.2021
##
#########################
##
##
##  This code contain plotting routines for the SWIRL class
##
#########################
#
## Imports
#
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm

#
#
def plot_decision(swirl, save=None):
# 
#********************
# Plot rho-delta and gamma decision diagrams 
#********************
##  INPUTS
#    
##  OUTPUTS
#
###############
  fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(5,5))

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

  # Colors
  Ndetections = len(swirl.vortices)

  ########################

  # Scatter
  axes[0].scatter(rho, delta, marker='o', color='gray', s=10)

  # Thresholds
  axes[0].axvline(rho_cutoff, linestyle=':', linewidth=1.0, color='orange')
  axes[0].axhline(delta_cutoff, linestyle=':', linewidth=1.0, color='orange')
  xgamma = np.linspace(0.001,1,500)
  ygamma = gamma_cutoff/xgamma
  axes[0].plot(xgamma, ygamma, linestyle='--', linewidth=1.0, color='orange')
  # Axes
  axes[0].set_xlim([0,1.05])
  axes[0].set_ylim([1e-3,2.0])
  axes[0].set_yscale('log')

  # Labels
  axes[0].set_xlabel(r'$\rho$')
  axes[0].set_ylabel(r'$\delta$')
  ########################

  # Scatter
  n = np.arange(gamma.shape[0])
  axes[1].plot(n,gamma, marker='o', markersize=2.7, color='gray', linestyle='none')

  # Thresholds
  axes[1].axhline(gamma_cutoff, linestyle='--', linewidth=1.0, color='orange')

  # Axes
  axes[1].set_yscale('log')
  axes[1].set_xlim([0,gamma.shape[0]*1.05])
  axes[1].set_ylim([1e-6,2.0])

  # Labels
  axes[1].set_xlabel(r'$n$')
  axes[1].set_ylabel(r'$\gamma$')

  #######################################

  # Figure pads
  fig.tight_layout(h_pad=0.5)

  if save:
    plt.savefig(save+'.png', dpi=200, bbox_inches = 'tight', pad_inches = 0.01, )
  plt.draw()
  fig.show()
##
##
#######################################################
#
#
def plot_evcmap(swirl, l=6, save=None):
# 
  N = swirl.v.x.shape[0]

  xrange = np.arange(0,N+1)
  yrange = np.arange(0,N+1)
  xgrid, ygrid = np.meshgrid(xrange,yrange)
  xgrid[...] = xgrid[...].T
  ygrid[...] = ygrid[...].T

  vx = swirl.v.x
  vy = swirl.v.y

  fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(5,8), sharex=True, sharey=True)

  ########################

  # Quiver plot
  axes[0].quiver(xgrid[::l,::l],ygrid[::l,::l],vx[::l,::l],vy[::l,::l], angles='xy', units='xy', scale=0.1, width=0.8)

  # Plot Criteria
  vmax = np.max(np.abs(swirl.R[0]))*1.0
  im0 = axes[0].imshow(swirl.R[0].T, origin='lower', cmap='PiYG', vmax=vmax, vmin=-vmax)

  # Axes
  axes[0].set_ylim([0,N])
  axes[0].set_xlim([0,N])
  axes[0].set_aspect('equal')
  axes[0].tick_params(axis="both", direction="in", which="both", right=True, top=True)
  axes[0].minorticks_on()

  # Colorbar
  divider = make_axes_locatable(axes[0])
  cax = divider.new_horizontal(size = '3%', pad = 0.03)
  fig.add_axes(cax)
  fig.colorbar(im0, cax = cax, orientation = 'vertical', label=r'Rortex $R$')

  ########################

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
  im1 = axes[1].scatter(xM, yM, s=10, c=sM, vmax=vmax, vmin=-vmax, cmap='Spectral_r')

  # Quiver plot
  axes[1].quiver(xgrid[::l,::l],ygrid[::l,::l],vx[::l,::l],vy[::l,::l], angles='xy', units='xy', scale=0.1, width=0.8)

  # Axes
  axes[1].set_ylim([0,N])
  axes[1].set_xlim([0,N])
  axes[1].set_aspect('equal')

  axes[1].tick_params(axis="both", direction="in", which="both", right=True, top=True)
  axes[1].minorticks_on()


  # Colorbar
  divider = make_axes_locatable(axes[1])
  cax = divider.new_horizontal(size = '3%', pad = 0.03)
  fig.add_axes(cax)
  fig.colorbar(im1, cax = cax, orientation = 'vertical', label=r'Grid cardinality $s$')

  #######################################

  # Figure pads
  fig.tight_layout(h_pad=0.00)

  if save:
    plt.savefig(save+'.png', dpi=200, bbox_inches = 'tight', pad_inches = 0.01, )

  plt.draw()
  fig.show()
##
##
#######################################################
#
#
def plot_vortices(swirl, l=6, save=None):
# 
  N = swirl.v.x.shape[0]
  
  xrange = np.arange(0,N+1)
  yrange = np.arange(0,N+1)
  xgrid, ygrid = np.meshgrid(xrange,yrange)
  xgrid[...] = xgrid[...].T
  ygrid[...] = ygrid[...].T

  vx = swirl.v.x
  vy = swirl.v.y

  fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(5,5), sharex=True, sharey=True)

  # Noise
  noisex = swirl.noise[2]
  noisey = swirl.noise[3]

  # Colors
  Ndetections = len(swirl.vortices)
  cMap = cm.get_cmap('PiYG', Ndetections)

  ########################

  ax = axes

  # Quiver plot
  ax.quiver(xgrid[::l,::l],ygrid[::l,::l],vx[::l,::l],vy[::l,::l], angles='xy', units='xy', scale=0.1, width=0.8)

  # Scatter Noise
  ax.scatter(noisex, noisey, marker='s', s=1, edgecolor='k', linewidth=1, facecolors='Gray', alpha=0.2, label=r'Noise')

  Ndetections = len(swirl.vortices)

  for i in np.arange(Ndetections):
    # Scatter Cells
    vcolor = np.mean(swirl.vortices[i].X)/220.+0.5
    if vcolor>0.5:
        vcolor=0.75
        vedge=0.9
    else:
        vcolor=0.25
        vedge=0.1
    circle = plt.Circle((swirl.vortices[i].center[0], 
                         swirl.vortices[i].center[1]), 
                        swirl.vortices[i].r, 
                        linewidth=0.3,
                        edgecolor=cMap(vedge), 
                        facecolor=cMap(vcolor), 
                        alpha=.3, 
                        label=r'Vortex '+str(i+1))

    # Scatter estimated centers
    ax.scatter(swirl.vortices[i].center[0], 
               swirl.vortices[i].center[1], 
               marker='*', 
               color=cMap(vedge), 
               s=80, 
               edgecolor='k', 
               linewidth=0.1)
    
    # Add circle
    ax.add_patch(circle)
    
          
                            
                  
  # Axes
  ax.set_ylim([0,N])
  ax.set_xlim([0,N])
  ax.set_aspect('equal')
  ax.tick_params(axis="both", direction="in", which="both", right=True, top=True)
  ax.minorticks_on()

  #######################################

  # Figure pads
  fig.tight_layout(h_pad=0.00)
  
  if save:
    plt.savefig(save+'.png', dpi=200, bbox_inches = 'tight', pad_inches = 0.01, )

  plt.draw()
  fig.show()