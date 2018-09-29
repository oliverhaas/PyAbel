# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import numpy as np
import abel
from scipy.linalg import inv
from scipy import dot

###############################################################################
#
#  Dasch two-point, three_point, and onion-peeling  deconvolution
#    as described in Applied Optics 31, 1146 (1992), page 1147-8 sect. B & C.
#        https://www.osapublishing.org/ao/abstract.cfm?uri=ao-31-8-1146
#    see also discussion in PR #155  https://github.com/PyAbel/PyAbel/pull/155
#
# 2016-03-25 Dan Hickstein - one line Abel transform
# 2016-03-24 Steve Gibson - Python code framework
# 2015-12-29 Dhrubajyoti Das - original three_point code and 
#                              highlighting the Dasch paper,see issue #61
#                              https://github.com/PyAbel/PyAbel/issues/61
#
###############################################################################

_dasch_parameter_docstring = \
    """dasch_method deconvolution
        C. J. Dasch Applied Optics 31, 1146 (1992).
        http://dx.doi.org/10.1364/AO.31.001146

    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    basis_dir: str
        path to the directory for saving / loading
        the "dasch_method" operator matrix.
        If None, the operator matrix will not be saved to disk.

    dr : float
        sampling size (=1 for pixel images), used for Jacobian scaling.
        The resulting inverse transform is simply scaled by 1/dr.

    direction: str
        only the `direction="inverse"` transform is currently implemented


    Returns
    -------
    inv_IM: 1D or 2D numpy array
        the "dasch_method" inverse Abel transformed half-image 

    """


def two_point_transform(IM, basis_dir='.', dr=1, direction="inverse"):
    return _dasch_transform(IM, basis_dir=basis_dir, dr=dr,
                            direction=direction, method="two_point")


def three_point_transform(IM, basis_dir='.', dr=1, direction="inverse"):
    return _dasch_transform(IM, basis_dir=basis_dir, dr=dr,
                            direction=direction, method="three_point")


def onion_peeling_transform(IM, basis_dir='.', dr=1, direction="inverse"):
    return _dasch_transform(IM, basis_dir=basis_dir, dr=dr,
                            direction=direction, method="onion_peeling")

two_point_transform.__doc__ =\
            _dasch_parameter_docstring.replace("dasch_method", "two-point")
three_point_transform.__doc__ =\
            _dasch_parameter_docstring.replace("dasch_method", "three-point")
onion_peeling_transform.__doc__ =\
            _dasch_parameter_docstring.replace("dasch_method", "onion-peeling")


def _dasch_transform(IM, basis_dir='.', dr=1, direction="inverse", 
                     method="three_point"):
    
    if direction != 'inverse':
        raise ValueError('Forward "two_point" transform not implemented')

    # make sure that the data has 2D shape
    IM = np.atleast_2d(IM)

    rows, cols = IM.shape

    if cols < 2 and method == "two_point":
        raise ValueError('"two_point" requires image width (cols) > 2')

    if cols < 3 and method == "three_point":
        raise ValueError('"three_point" requires image width (cols) > 3')
    
    D = abel.tools.basis.get_bs_cached(method, cols, basis_dir=basis_dir)

    inv_IM = dasch_transform(IM, D)

    if rows == 1:
        inv_IM = inv_IM[0]  # flatten array

    return inv_IM/dr


def dasch_transform(IM, D):
    """Inverse Abel transform using a given D-operator basis matrix.

    Parameters
    ----------
    IM : 2D numpy array
        image data
    D : 2D numpy array 
        D-operator basis shape (cols, cols) 

    Returns
    -------
    inv_IM : 2D numpy array
        inverse Abel transform according to basis operator D 
    """
    # one-line Abel transform - dot product of each row of IM with D
    return np.tensordot(IM, D, axes=(1, 1))


def _bs_two_point(cols):
    """basis function for two_point.
    
    Parameters
    ----------
    cols : int
        width of the image
    """

    # basis function Eq. (9)  for j >= i
    def J(i, j): 
        return np.log((np.sqrt((j+1)**2 - i**2) + j + 1)/
                      (np.sqrt(j**2 - i**2) + j))/np.pi

    # Eq. (8, 9) D-operator basis, is 0 for j < i
    D = np.zeros((cols, cols))

    # diagonal i == j
    Ii, Jj = np.diag_indices(cols) 
    Ii = Ii[1:]  # exclude special case i=j=0
    Jj = Jj[1:]
    D[Ii, Jj] = J(Ii, Jj)

    # upper triangle j > i
    Iu, Ju = np.triu_indices(cols, k=1)
    Iu = Iu[1:]  # exclude special case [0, 1]
    Ju = Ju[1:]
    D[Iu, Ju] = J(Iu, Ju) - J(Iu, Ju-1)

    # special cases
    D[0, 1] = J(0, 1) - 2/np.pi
    D[0, 0] = 2/np.pi

    return D


def _bs_three_point(cols):
    """basis function for three_point.
    
    Parameters
    ----------
    cols : int
        width of the image

    modified by Oliver Haas (github.com/oliverhaas) to be consistently second order
    approximation and deal with non-zero upper end of the data.
    """

    # finite difference derivative stencil
    FD = np.zeros((cols,cols))
    FD[0,0:3] = [-1.5, 2., -0.5]
    FD[-1,-3::] = [0.5, -2., 1.5]
    I = np.arange(cols)
    FD[I[1:-1],I[:-2]] = -0.5
    FD[I[1:-1],I[2:]] = 0.5

    # finite difference second derivative stencil
    FD2 = np.zeros((cols,cols))
    FD2[0,0:4] = [2., -5., 4., -1.]
    FD2[-1,-4::] = [-1., 4., -5., 2.]
    I = np.arange(cols)
    FD2[I[1:-1],I[:-2]] = 1.
    FD2[I[1:-1],I[1:-1]] = -2.
    FD2[I[1:-1],I[2:]] = 1.

    # Analytical integration coefficients
    def c0(i, j):
        return np.log(((np.sqrt((2*j + 1)**2 - 4*i**2) + 2*j + 1))/ 
                       (np.sqrt((2*j - 1)**2 - 4*i**2) + 2*j - 1))

    def c1(i, j):
        return 0.5*(np.sqrt((2*j+1)**2 - 4*i**2) - np.sqrt((2*j-1)**2 - 4*i**2)) - j*c0(i,j)

    def c0diag(i):
        return np.log((np.sqrt(1 + 4*i) + 2*i + 1)/(2*i))

    def c1diag(i):
        return 0.5*np.sqrt(1 + 4*i) - i*c0diag(i)

    def c0end(i, j):
        return np.log((2*np.sqrt(j**2 - i**2) + 2*j)/ 
                      (np.sqrt((2*j - 1)**2 - 4*i**2) + 2*j - 1))

    def c1end(i, j):
        return (np.sqrt(j**2 - i**2) - 0.5*np.sqrt((2*j-1)**2 - 4*i**2)) - j*c0end(i,j)


    # Analytical integration matrix
    D0 = np.zeros((cols,cols))
    D1 = np.zeros((cols,cols))

    # diagonal
    I = np.arange(1, cols-1)
    D0[I,I] = c0diag(I)
    D1[I,I] = c1diag(I)
    D0[0,0] = 0.
    D1[0,0] = 0.5

    # triangle bulk
    Iut, Jut = np.triu_indices(cols-1, k=1)
    D0[Iut, Jut] = c0(Iut,Jut)
    D1[Iut, Jut] = c1(Iut,Jut)

    # end
    Iend = np.arange(cols-1)
    D0[Iend,-1] = c0end(Iend,cols-1)
    D1[Iend,-1] = c1end(Iend,cols-1)

    D = -(D0.dot(FD) + D1.dot(FD2))/np.pi

    return D


def _bs_onion_peeling(cols):
    """basis function for onion_peeling.
    
    Parameters
    ----------
    cols : int
        width of the image
    """

    # basis weight matrix 
    W = np.zeros((cols, cols))

    # diagonal elements i = j, Eq. (11)
    I, J = np.diag_indices(cols) 
    W[I, J] = np.sqrt((2*J+1)**2 - 4*I**2)

    # upper triangle j > i,  Eq. (11)
    Iu, Ju = np.triu_indices(cols, k=1) 
    W[Iu, Ju] = np.sqrt((2*Ju + 1)**2 - 4*Iu**2) -\
                np.sqrt((2*Ju - 1)**2 - 4*Iu**2) 

    # operator used in Eq. (1)
    D = inv(W)   

    return D
