#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This example compares the available inverse Abel transform methods
# for the Ominus sample image
#
# Note it transforms only the Q0 (top-right) quadrant
# using the fundamental transform code

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel

import collections
import matplotlib.pylab as plt
from time import time

# inverse Abel transform methods -----------------------------
#   dictionary of method: function()

transforms = {
  "direct": abel.direct.direct_transform,      
  #"onion": iabel_onion_transform, 
  "hansenlaw": abel.hansenlaw.hansenlaw_transform,
  "basex": abel.basex.basex_transform,   
  "three_point": abel.three_point.three_point_transform,
}
# sort dictionary 
transforms = collections.OrderedDict(sorted(transforms.items()))
ntrans = np.size(transforms.keys())  # number of transforms


# Image:   O2- VMI 1024x1024 pixel ------------------
IM = abel.tools.analytical.sample_image(n=1001, name="dribinski")

h, w = IM.shape

# forward transform
fIM = abel.transform(IM, direction="forward", method="hansenlaw")['transform']

Q0, Q1, Q2, Q3 = abel.tools.symmetry.get_image_quadrants(fIM, reorient=True)

Q0fresh = Q0.copy()    # keep clean copy
print ("quadrant shape {}".format(Q0.shape))

# process Q0 quadrant using each method --------------------

iabelQ = []  # keep inverse Abel transformed image

for q, method in enumerate(transforms.keys()):

    Q0 = Q0fresh.copy()   # top-right quadrant of O2- image

    print ("\n------- {:s} inverse ...".format(method))  
    t0 = time()

    # inverse Abel transform using 'method'
    IAQ0 = transforms[method](Q0, direction="inverse") 

    print ("                    {:.1f} sec".format(time()-t0))

    iabelQ.append(IAQ0)  # store for plot

    # polar projection and speed profile
    radial, speed = abel.tools.vmi.angular_integration(IAQ0, origin=(0, 0), Jacobian=False)

    # normalize image intensity and speed distribution
    IAQ0 /= IAQ0.max()  
    speed /= speed.max()

    # plots    #121 whole image,   #122 speed distributions
    plt.subplot(121) 

    # method label for each quadrant
    annot_angle = -(45+q*90)*np.pi/180  # -ve because numpy coords from top
    if q > 3: 
        annot_angle += 50*np.pi/180    # shared quadrant - move the label  
    annot_coord = (h/2+(h*0.9)*np.cos(annot_angle)/2, 
                   w/2+(w*0.9)*np.sin(annot_angle)/2)
    plt.annotate(method, annot_coord, color="yellow")

    # plot speed distribution
    plt.subplot(122) 
    plt.plot(radial, speed, label=method)

# reassemble image, each quadrant a different method

# for < 4 images pad using a blank quadrant
blank = np.zeros(IAQ0.shape)  
for q in range(ntrans, 4):
    iabelQ.append(blank)

# more than 4, split quadrant
if ntrans == 5:
    # split last quadrant into 2 = upper and lower triangles
    tmp_img = np.tril(np.flipud(iabelQ[-2])) +\
              np.triu(np.flipud(iabelQ[-1]))
    iabelQ[3] = np.flipud(tmp_img)
# Fix me when > 5 images
 

im = abel.tools.symmetry.put_image_quadrants((iabelQ[0], iabelQ[1],
                                              iabelQ[2], iabelQ[3]), 
                                              odd_size=True)

plt.subplot(121)
plt.imshow(im, vmin=0, vmax=0.1)

plt.subplot(122)
plt.title("dribinski sample image")
plt.axis(ymin=-0.05, ymax=0.5, xmin=0, xmax=200)
plt.legend(loc=0, labelspacing=0.1)
plt.tight_layout()
plt.savefig('example_all_dribinski.png', dpi=100)
plt.show()
