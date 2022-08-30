"""
SWIRL Code
    evcmap.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains the routines needed to compute
the EVC map based on the estimated vortex center method.
"""
# Imports
import numpy as np
# ----------------


def compute_evcmap(rortex, vgt, v, grid_dx):
    """
    Computes the EVC map arrays for all the stencils given the
    mathematical criterion, the velocity field and its gradients, and
    the cell sizes.

    Parameters
    ----------
    rortex : list of arrays
        Mathematical criterion arrays
    vgt : list of arrays
        Velocity gradient tensor arrays
    v : 2D vector
        Velocity field.
    grid_dx : 2D vector
        Grid cell size

    Returns
    -------
    M : list of arrays
        G-EVC maps.
    dataCells : list of arrays
        array containing original coordinates, EVC coordinates,
        and criteria values.
    noise : array
        An array containing the cells identified as noise because isolated
        in the G-EVC map.
    Raises
    ------
    """
    # Initialize empty lists
    dataCells = []

    # get size of grid
    nx = rortex[0].shape[0]
    ny = rortex[0].shape[1]

    # Initialize the cardinality of EVC map
    S = np.zeros((nx, ny))

    # Initialize noise
    noise = np.zeros((2,0))
    #print('0.\n',noise)

    # loop over stencils:
    for n in np.arange(len(rortex)):

        # build grids
        x, y = np.meshgrid(np.arange(0, nx), np.arange(0, ny), indexing='ij')

        
        # find grid points where rortex != 0 (i.e where curvature is present)
        #curvature_mask = np.where(np.abs(rortex[n]) > 0.0)
        #x = curvature_mask[0]
        #y = curvature_mask[1]
        

        # compute estimated radius (in units)
        r = radius(rortex[n], v)

        # compute estimated center direction
        ex, ey = direction(vgt[n], v)

        # compute jump in pixels to reach
        # center from coordinates
        px = r*ex/grid_dx.x
        py = r*ey/grid_dx.y

        # Build center maps
        xc = x + px
        yc = y + py
        # xc = x + px[curvature_mask]
        # yc = y + py[curvature_mask]

        

        # # Flatten lists of coordinates
        x = x.flatten()
        y = y.flatten()
        xc = xc.flatten()
        yc = yc.flatten()
        r = r.flatten()
        Xi = rortex[n].flatten()

        # Remove evcs outside of domain
        xc = np.where(xc > nx, -1., xc)
        xc = np.where(xc < 0., -1., xc)
        yc = np.where(yc > ny, -1., yc)
        yc = np.where(yc < 0., -1., yc)
        xc = np.where(rortex[n].ravel() == 0., -2., xc)
        yc = np.where(rortex[n].ravel() == 0., -2., yc)

        outside_mask = np.where(xc == -2.0)
        x = np.delete(x, outside_mask)
        y = np.delete(y, outside_mask)
        xc = np.delete(xc, outside_mask)
        yc = np.delete(yc, outside_mask)
        r = np.delete(r, outside_mask)
        Xi = np.delete(Xi, outside_mask)
        outside_mask = np.where(yc == -2.0)
        x = np.delete(x, outside_mask)
        y = np.delete(y, outside_mask)
        xc = np.delete(xc, outside_mask)
        yc = np.delete(yc, outside_mask)
        r = np.delete(r, outside_mask)
        Xi = np.delete(Xi, outside_mask)

        outside_mask = np.where(xc == -1.0)
        noise_n = np.array([x[outside_mask],y[outside_mask]])
        noise = np.concatenate((noise, noise_n), axis=1)
        x = np.delete(x, outside_mask)
        y = np.delete(y, outside_mask)
        xc = np.delete(xc, outside_mask)
        yc = np.delete(yc, outside_mask)
        r = np.delete(r, outside_mask)
        Xi = np.delete(Xi, outside_mask)
        outside_mask = np.where(yc == -1.0)
        noise_n = np.array([x[outside_mask],y[outside_mask]])
        noise = np.concatenate((noise, noise_n), axis=1)
        x = np.delete(x, outside_mask)
        y = np.delete(y, outside_mask)
        xc = np.delete(xc, outside_mask)
        yc = np.delete(yc, outside_mask)
        r = np.delete(r, outside_mask)
        Xi = np.delete(Xi, outside_mask)

        # Compute GEVC map
        grid_xc = np.round(xc)
        grid_yc = np.round(yc)
        weight = np.sign(Xi)
        # Compute number of points per cell with np.histogram
        Si, _, _ = np.histogram2d(grid_xc, grid_yc, weights=weight, bins=[
                                  np.arange(0, nx+1), np.arange(0, ny+1)])
        # Remove single counters
        single_gevcs = np.where(np.abs(Si) == 1.0)
        Si[single_gevcs] = 0.

        grid_single = single_gevcs[0] + 1j*single_gevcs[1]
        grid_c = grid_xc + 1j*grid_yc
        single_mask = np.where(np.in1d(grid_c, grid_single))[0]
        # print(single_mask)
        noise_n = np.array([x[single_mask], y[single_mask]])
        noise = np.concatenate((noise, noise_n), axis=1)
        x = np.delete(x, single_mask)
        y = np.delete(y, single_mask)
        xc = np.delete(xc, single_mask)
        yc = np.delete(yc, single_mask)
        r = np.delete(r, single_mask)
        Xi = np.delete(Xi, single_mask)
        

        # # Remove evcs outside of domain an place x,y coordinates
        # # into noise
        # outside_mask = np.where(xc > nx)
        # noise_n = np.array([x[outside_mask],y[outside_mask]])
        # noise = np.concatenate((noise, noise_n), axis=1)
        # x = np.delete(x, outside_mask)
        # y = np.delete(y, outside_mask)
        # xc = np.delete(xc, outside_mask)
        # yc = np.delete(yc, outside_mask)


        # outside_mask = np.where(xc < 0.)
        # noise_n = np.array([x[outside_mask],y[outside_mask]])
        # noise = np.concatenate((noise, noise_n), axis=1)
        # x = np.delete(x, outside_mask)
        # y = np.delete(y, outside_mask)
        # xc = np.delete(xc, outside_mask)
        # yc = np.delete(yc, outside_mask)


        # outside_mask = np.where(yc > ny)
        # noise_n = np.array([x[outside_mask],y[outside_mask]])
        # noise = np.concatenate((noise, noise_n), axis=1)
        # x = np.delete(x, outside_mask)
        # y = np.delete(y, outside_mask)
        # xc = np.delete(xc, outside_mask)
        # yc = np.delete(yc, outside_mask)


        # outside_mask = np.where(yc < 0.)
        # noise_n = np.array([x[outside_mask],y[outside_mask]])
        # noise = np.concatenate((noise, noise_n), axis=1)
        # x = np.delete(x, outside_mask)
        # y = np.delete(y, outside_mask)
        # xc = np.delete(xc, outside_mask)
        # yc = np.delete(yc, outside_mask)

        # # Compute GEVC map
        # grid_xc = np.array(xc, dtype=int)
        # grid_yc = np.array(yc, dtype=int)
        # rortex_c = rortex[n][grid_xc, grid_yc]
        # weight = np.sign(rortex_c)
        # # Compute number of points per cell with np.histogram
        # Si, _, _ = np.histogram2d(grid_xc, grid_yc, weights=weight, bins=[
        #                           np.arange(0, nx+1), np.arange(0, ny+1)])
        # # Remove single counters
        # single_gevcs = np.where(np.abs(Si) == 1.0)
        # grid_single = single_gevcs[0] + 1j*single_gevcs[1]
        # grid_c = grid_xc + 1j*grid_yc
        # single_mask = np.where(np.in1d(grid_c, grid_single))[0]
        # #print(single_mask)
        # noise_n = np.array([x[single_mask], y[single_mask]])
        # noise = np.concatenate((noise, noise_n), axis=1)
        # x = np.delete(x, single_mask)
        # y = np.delete(y, single_mask)
        # xc = np.delete(xc, single_mask)
        # yc = np.delete(yc, single_mask)
        # Si[single_gevcs] = 0.

        """
        # Treat outside points
        # Flag ecvs outside of domain
        xc = np.where(xc > nx, -1., xc)
        xc = np.where(xc < 0., -1., xc)
        yc = np.where(yc > ny, -1., yc)
        yc = np.where(yc < 0., -1., yc)

        # Flag ecvs where criterion is null
        xc = np.where(rortex[n] == 0., -2., xc)
        yc = np.where(rortex[n] == 0., -2., yc)
        #print(xc)
        #print(yc)
        # Flatten all quantities
        x = x.flatten()
        y = y.flatten()
        xc = xc.flatten()
        yc = yc.flatten()
        r = r.flatten()
        Xi = rortex[n].flatten()

        # Remove the flagged points (where R=0)
        mask = np.where(xc == -2.)
        xc = np.delete(xc, mask)
        yc = np.delete(yc, mask)
        x = np.delete(x, mask)
        y = np.delete(y, mask)
        r = np.delete(r, mask)
        Xi = np.delete(Xi, mask)
        mask = np.where(yc == -2.)
        xc = np.delete(xc, mask)
        yc = np.delete(yc, mask)
        x = np.delete(x, mask)
        y = np.delete(y, mask)
        r = np.delete(r, mask)
        Xi = np.delete(Xi, mask)

        # Remove the flagged points
        mask = np.where(xc == -1.)
        noise_n = np.array((x[mask],
                            y[mask]
                            ))
        noise = np.concatenate((noise, noise_n), axis=1)
        xc = np.delete(xc, mask)
        yc = np.delete(yc, mask)
        x = np.delete(x, mask)
        y = np.delete(y, mask)
        r = np.delete(r, mask)
        Xi = np.delete(Xi, mask)
        ##print('1.\n',noise)

        mask = np.where(yc == -1.)
        noise_n = np.array((x[mask],
                            y[mask]
                            ))
        noise = np.concatenate((noise, noise_n), axis=1)
        xc = np.delete(xc, mask)
        yc = np.delete(yc, mask)
        x = np.delete(x, mask)
        y = np.delete(y, mask)
        r = np.delete(r, mask)
        Xi = np.delete(Xi, mask)
        #print('2.\n',noise)
        
        # Define coordinate of cell centers and weights
        xcell = np.round(xc)
        ycell = np.round(yc)
        weight = np.sign(Xi)

        # Compute number of points per cell with np.histogram
        Si, _, _ = np.histogram2d(xcell, ycell, weights=weight, bins=[
                                  np.arange(0, nx+1), np.arange(0, ny+1)])

        # Remove single counters
        mask = np.where(np.abs(Si) < 2.0)
        noise_n = np.array((mask[0], 
                            mask[1]
                            ))
        #noise = np.concatenate((noise, noise_n), axis=1)
        #print('3.\n',noise)
        Si[mask] = 0.
        """
        

        # Add to cardinality map
        S = S + Si

        # Fill dataCells
        dataCells_n = np.zeros((7, xc.shape[0]))

        
        # grid_x = np.array(x, dtype=int)
        # grid_y = np.array(y, dtype=int)
        # Xi = rortex[n][grid_x, grid_y]
        # r = r[grid_x, grid_y]
        

        dataCells_n[0] = xc  # EVC coord x
        dataCells_n[1] = yc  # EVC coord y
        dataCells_n[2] = x  # Cell coord x
        dataCells_n[3] = y  # Cell coord y
        dataCells_n[4] = Xi  # Criterium value
        dataCells_n[5] = r  # Estimated distance from vortex center
        dataCells_n[6] = n*np.ones((xc.shape[0]))  # Stencil index

        # Append
        dataCells.append(dataCells_n)

    # New grid of coordinates
    xgrid, ygrid = np.meshgrid(
        np.arange(0, nx), np.arange(0, ny), indexing='ij')

    # Flatten
    S = S.flatten()
    xgrid = xgrid.flatten()
    ygrid = ygrid.flatten()

    # Find coordinates and number of centers
    # Remove point where only one or less ecv are found
    mask = np.where(np.abs(S) < 2.)
    S = np.delete(S, mask)
    xgrid = np.delete(xgrid, mask)
    ygrid = np.delete(ygrid, mask)

    # Assembling the output data arranged in ascending order
    M = np.zeros((3, S.shape[0]))
    inds = S.argsort()
    M[0] = xgrid[inds]
    M[1] = ygrid[inds]
    M[2] = S[inds]

    # Make array
    dataCells = np.hstack(dataCells)

    return M, dataCells, noise
# ---------------------


def radius(rortex, v):
    """
    Computes the radius of curvature for all the cells where
    the flow appears to be curved (mathematical criteria != 0).

    Parameters
    ----------
    rortex : array
        Mathematical criterion array
    v : 2D vector
        Velocity field.

    Returns
    -------
    r : array
        Curvature radius in physical units.

    Raises
    ------
    """
    # compute vnorm
    vnorm = np.sqrt(v.x**2 + v.y**2)

    # radius in units
    # ! avoid rortex = 0.0 points
    with np.errstate(divide='ignore'):
        r = np.where(np.abs(rortex) == 0.,  0., 2.*vnorm/np.abs(rortex))

    return r
# ----------


def direction(vgt, v):
    """
    Computes the radial direction for all the cells where
    the flow appears to be curved (mathematical criteria != 0).

    Parameters
    ----------
    vgt : list of arrays
        Velocity gradient tensor arrays
    v : 2D vector
        Velocity field.

    Returns
    -------
    ex : array
        The x radial direction (normalized).
    ey : array
        The y radial direction (normalized).

    Raises
    ------
    """
    # invert vgt
    # ! all non invertible points already treated in swirling strength
    vgt_inv = np.linalg.inv(vgt)

    # compute directions
    ex = -(vgt_inv[:, :, 0, 0]*v.x + vgt_inv[:, :, 0, 1]*v.y)
    ey = -(vgt_inv[:, :, 1, 0]*v.x + vgt_inv[:, :, 1, 1]*v.y)

    # normalize
    enorm = np.sqrt(ex**2 + ey**2)
    ex = np.where(enorm > 0., ex/enorm, 0.)
    ey = np.where(enorm > 0., ey/enorm, 0.)

    return ex, ey
