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
from scipy import ndimage


def meandist2points(image, kpoints):
    length = np.max(np.shape(image))
    dists =metrics.pairwise.euclidean_distances(image, kpoints)**2
    nearest = np.argmin(dists, axis =1)
    vecdists = []
    for pixel in range(length):
        val = image[pixel, :] - kpoints[nearest[pixel],:] + 0.0
        vecdists.append(np.sqrt(val.dot(val)))
        #sum_of_dists += np.sqrt(val.dot(val))
    #return sum_of_dists/length
    return np.mean(vecdists)


img = ndimage.imread('lola.JPG')

img_height = np.shape(img)[0]
img_width = np.shape(img)[1]

plt.imshow(img)

new_img = np.reshape(img, (img_width*img_height, 3))

num_ks = 20
for k in range(14, num_ks):
    kvec = np.random.uniform(0, 255, (k, 3))
    err = meandist2points(new_img, kvec)
    while True:
        dists =metrics.pairwise.euclidean_distances(new_img, kvec)**2
    
        nearest = np.argmin(dists, axis =1)
        #bestdists = np.min(dists,axis =1)
        #avg_dists = np.mean(bestdists)
        #print avg_dists
        #print  meandist2points(new_img, kvec)
    
        #print kvec
        for means in range(k):
            new_point = np.mean(new_img[nearest== means, :],axis= 0)
            if new_point[0] != new_point[0]: # nan because no pixels are with this point
                new_point = np.random.uniform(0, 255, (1, 3))          
            #new_point = np.nan_to_num(np.mean(new_img[nearest== means, :],axis= 0))
            #if new_point[0] < 1 and new_point[1] < 1 and new_point[2] < 1:
            #    new_point = np.random.uniform(0, 255, (1, 3))
            kvec[means, :] = new_point
    
        err2 = meandist2points(new_img, kvec)
        print err2
        print ''
        if np.abs(err2 - err) < .01:
            break
        err = err2
    
    
    
    kfinal = np.round(kvec)
    dists =metrics.pairwise.euclidean_distances(new_img, kfinal)
    
    nearest = np.argmin(dists, axis =1)
    
    #print kvec
    compressed_img = np.zeros((img_width*img_height, 3))
    #compressed_img[:] = new_img
    for means in range(k):
        compressed_img[nearest == means, :] = kfinal[means]
    
    final_img = np.reshape(compressed_img, (img_height,img_width,3))
    plt.imshow(-final_img)
    
    stringname = 'lola%dcolor.png' % k
    plt.imsave(fname=stringname, arr= -final_img, format='png')

#residual = np.abs(compressed_img - new_img)
#residual = np.sum(residual, axis = 1)
#plt.hist(residual, 20)

#plt.imshow(final_img)


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(kvec[:,0], kvec[:, 1], kvec[:, 2]  )
#plt.show()

