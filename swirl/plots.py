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
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm

#
#
def plot_decision(swirl):
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
    axes[0].set_ylim([1e-3,1.05])
    axes[0].set_yscale('log')

    # Labels
    axes[0].set_xlabel(r'$\rho$')
    axes[0].set_ylabel(r'$\delta$')
    ########################

    # Scatter
    n = np.arange(gamma.shape[0])
    axes[1].plot(n[:-2],gamma[:-2], marker='o', markersize=2.7, color='gray', linestyle='none')

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

    plt.draw()
    fig.show()

  ##
  ##
  ###########################################