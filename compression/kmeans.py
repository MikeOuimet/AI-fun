# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:23:10 2016
fun with compression

@author: mike
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics



img = np.load('frame.npy')
new_img = np.reshape(img, (480*640, 3))

k = 16

kvec = np.random.uniform(0, 255, (k, 3))

for step in range(10):
    print step
    dists =metrics.pairwise.euclidean_distances(new_img, kvec)**2

    nearest = np.argmin(dists, axis =1)

    print kvec
    for means in range(k):
        new_point = np.nan_to_num(np.mean(new_img[nearest== means, :],axis= 0))
        if new_point[0] == 0:
            new_point = np.random.uniform(0, 255, (1, 3))
        kvec[means, :] = new_point


#npoints = 50000
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kvec[:,0], kvec[:, 1], kvec[:, 2]  )
plt.show()

