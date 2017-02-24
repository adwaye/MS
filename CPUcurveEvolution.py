# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:32:01 2017

@author: amr62
"""

import os


os.chdir("Documents/TheEffingPhDHatersGonnaHate/fastmsToying/trialimages/")


import myCudaModule as mcm
import linAlgebraModule as linalg
import scipy.misc as scm
import scipy.ndimage as scnd
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as skex
import time
import pycuda.driver as drv
import pycuda.tools as tl
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import sys
import time

# setting up the initial condition, f_0 = dist, signed distance transform of a curve in image
image = scm.imread('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/fastmsToying/trialimages/curve.jpg').astype(np.float32)
image = mcm.grayfication(image)
image[image.nonzero()]=1




#finding interior of curve

def curveInterior(image):
    forme = image.shape
    if(np.size(forme)>2):
        sys.exit('Only works on gray images')


    aSize = forme[0]*forme[1]
    xdim  = np.int32(forme[0])
    ydim  = np.int32(forme[1])   
    labIm = 1*image    
    for i in range(0,xdim):
        arr     = (1-image[i,:]).nonzero()
        if(np.size(arr)!=0):        
            maxind  = np.max(arr)
            minind  = np.min(arr)
            labIm[i,:][minind:maxind] = 0
    
    for j in range(0,ydim):
        arr     = (1-image[:,j]).nonzero()
        if(np.size(arr)!=0):        
            maxind  = np.max(arr)
            minind  = np.min(arr)
            labIm[:,j][minind:maxind] = 0
                        
    return labIm
                    



dist                       = scnd.morphology.distance_transform_edt(image).astype(np.float32)
labIm                      = curveInterior(image)
dist[(1-labIm).nonzero()]  = -dist[(1-labIm).nonzero()]
#np.loadtxt('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/fastmsToying/trialimages/distance.dat',delimiter=',').astype(np.float32)
dist = dist/(np.max(dist)-np.min(dist))
#dist = scnd.gaussian_filter(dist,4)

#dist = scm.imresize(dist,0.5)
#


maxit    = 100
timestep = 0.001


xdim = dist.shape[0]
ydim = dist.shape[1]

shupright   = np.zeros((xdim,ydim))
shupleft    = np.zeros((xdim,ydim))
shdownright = np.zeros((xdim,ydim))
shdownleft  = np.zeros((xdim,ydim))

gradx, grady = np.gradient(dist)
maingrad     = np.concatenate( ( gradx.reshape((xdim,ydim,1)) , grady.reshape((xdim,ydim,1))  ), axis= 2   )

gradXX, gradXY, gradYY = np.gradient(maingrad)



for k in range(0,maxit):
    shupright[0:(xdim-2),0:(ydim-2)]    = dist[1:(xdim-1),1:(ydim-1)] #shdupright(i,j)  = image(i+1,j+1)
    shupleft[1:(xdim-1),0:(ydim-2)]     = dist[0:(xdim-2),1:(ydim-1)] # shupleft(i,j)   = image(i-1,j+1)
    shdownright[0:(xdim-2),1:(ydim-1)]  = dist[1:(xdim-1),0:(ydim-2)] #shdownright(i,j) = image(i+1,j-1)
    shdownleft[1:(xdim-1),1:(ydim-1)]   = dist[0:(xdim-2),0:(ydim-2)] # shdownleft(i,j)  = image(i-1,j-1)   
    gradx, grady = np.gradient(dist)
    
    gradxx = sh

colmap = 'Greys'
fig2 = plt.figure(1)

ax0 = fig2.add_subplot(221)
ax0.imshow(gradx,cmap=colmap)
ax0.set_title('gradx')
ax0.axis('off')

ax1 = fig2.add_subplot(222)
ax1.imshow(grady,cmap=colmap)
ax1.set_title('grady')
ax1.axis('off')


ax2 = fig2.add_subplot(223)
ax2.imshow(dist,cmap=colmap)
ax2.set_title('dist')
ax2.axis('off')




