# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import matplotlib.pyplot as plt


# Dribinski sample image
IM = abel.tools.analytical.SampleImage(n=601).image 

# split into quadrants
origQ = abel.tools.symmetry.get_image_quadrants(IM)

# speed distribution
orig_speed = abel.tools.vmi.angular_integration(origQ[0], origin=(0,0))

# forward Abel projection
fIM = abel.Transform(IM, direction="forward", method="openAbel", 
                     transform_options={'method':3, 'order':2}).transform

# split projected image into quadrants
Q = abel.tools.symmetry.get_image_quadrants(fIM)
Q0 = Q[0].copy()

# openAbel inverse Abel transform
borQ0 = abel.openAbel.openAbel_transform(Q0, method = 3, order = 2)
# speed distribution
bor_speed = abel.tools.vmi.angular_integration(borQ0, origin=(0,0))

plt.plot(*orig_speed, linestyle='dashed', label="Dribinski sample")
plt.plot(bor_speed[0], bor_speed[1], label="openAbel")
plt.axis(ymin=-0.1)
plt.legend(loc=0)
plt.savefig("plot_example_openAbel.png",dpi=100)
plt.show()
