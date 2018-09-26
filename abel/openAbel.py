# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage import interpolation

import openAbel

#############################################################################
# hansenlaw - a recursive method forward/inverse Abel transform algorithm
#
# Adapted from (see also PR #211):
#  [1] E. W. Hansen "Fast Hankel Transform"
#      IEEE Trans. Acoust. Speech, Signal Proc. 33(3), 666-671 (1985)
#      doi: 10.1109/TASSP.1985.1164579
#
# and:
#
#  [2] E. W. Hansen and P-L. Law
#      "Recursive methods for computing the Abel transform and its inverse"
#      J. Opt. Soc. Am A2, 510-520 (1985)
#      doi: 10.1364/JOSAA.2.000510
#
# 2018-04   : New code rewrite, implementing the 1st-order hold approx. of
#             Ref. [1], with the assistance of Eric Hansen. See PR #211.
#
#             Original hansenlaw code was based on Ref. [2]
#
# 2018-03   : NB method applies to grid centered (even columns), not
#             pixel-centered (odd column) image see #206, #211
#             Apply, -1/2 pixel shift for odd column full image
# 2018-02   : Drop one array dimension, use numpy broadcast multiplication
# 2015-12-16: Modified to calculate the forward Abel transform
# 2015-12-03: Vectorization and code improvements Dan Hickstein and
#             Roman Yurchak
#             Previously the algorithm iterated over the rows of the image
#             now all of the rows are calculated simultaneously, which provides
#             the same result, but speeds up processing considerably.
#
# Historically, this algorithm was adapted by Jason Gascooke from ref. [2] in:
#
#  J. R. Gascooke PhD Thesis:
#   "Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals
#    Molecule Dissociation", Flinders University, 2000.
#
# Implemented in Python, with image quadrant co-adding, by Stephen Gibson (ANU)
# Significant code/speed improvements due to Dan Hickstein and Roman Yurchak
#
# Stephen Gibson - Australian National University, Australia
#
#############################################################################


def openAbel_transform(image, dr=1, direction='inverse', **kwargs):
    r"""Wrapper for the openAbel implementations of Abel transforms.


    This function performs the transform on only one "right-side"
    image. ::

    .. note::  Image should be a right-side image, like this: ::

        .         +--------      +--------+
        .         |      *       | *      |
        .         |   *          |    *   |  <---------- im
        .         |  *           |     *  |
        .         +--------      o--------+
        .         |  *           |     *  |
        .         |   *          |    *   |
        .         |     *        | *      |
        .         +--------      +--------+

        In accordance with all PyAbel methods the image center ``o`` is
        defined to be mid-pixel i.e. an odd number of columns, for the
        full image.


    For the full image transform, use the :class:``abel.Transform``.

    Inverse Abel transform: ::

      iAbel = abel.Transform(image, method='openAbel').transform

    Forward Abel transform: ::

      fAbel = abel.Transform(image, direction='forward', method='openAbel').transform


    Parameters
    ----------
    image : 1D or 2D numpy array
        Right-side half-image (or quadrant). See figure below.

    dr : float
        Sampling size, used for Jacobian scaling.
        Default: `1` (appliable for pixel images).

    direction : string 'forward' or 'inverse'
        ``forward`` or ``inverse`` Abel transform.
        Default: 'inverse'.


    Returns
    -------
    aim : 1D or 2D numpy array
        forward/inverse Abel transform half-image


    """

    image = np.atleast_2d(image)   # 2D input image
    aim = np.empty_like(image)  # Abel transform array
    rows, cols = image.shape

    if direction == 'forward':
        forwardBackward = -1
    else:
        forwardBackward = 1

    if kwargs.get('method') == None:
        method = 3
    else:
        method = kwargs.get('method')

    if kwargs.get('order') == None:
        order = 2
    else:
        order = kwargs.get('order')

    try:
        abelObj = openAbel.Abel(cols, forwardBackward, 0., dr, method = method, order = order)
        for ii in range(rows):
            aim[ii,:] = abelObj.execute(image[ii,:])
    except:
        raise

    if rows == 1:
        aim = aim[0]  # flatten to a vector

    return aim
