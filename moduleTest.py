# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:49:19 2017

@author: amr62
"""
#boundary values of gradient is shit
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
"""
image    = scm.imread('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/fastmsToying/trialimages/546h2016_1_0.jpg').astype(np.float32)

grayimage = mcm.grayfication(image)

minval = 80
maxval = 200

thresholdedimage = mcm.threshold(grayimage,minval, maxval)


#gradx, grady = mcm.grad(grayimage)
grayimage              = dist 
gradx, grady           = mcm.grad(grayimage)
gradx1, grady1         = np.gradient(grayimage)
gradxx, gradxy, gradyy = mcm.secOrderGrad(grayimage)
div                    = mcm.anDiffOperator(grayimage)

colmap = 'Greys'
fig = plt.figure(2)
ax0 = fig.add_subplot(221)
ax0.imshow(gradx1,cmap=colmap)
ax0.set_title('gradx np')
ax0.axis('off')

ax1 = fig.add_subplot(222)
ax1.imshow(grady1,cmap=colmap)
ax1.set_title('grady np')
ax1.axis('off')

ax2 = fig.add_subplot(223)
ax2.imshow(gradx[1:598,:],cmap=colmap)
ax2.set_title('gradx gpu')
ax2.axis('off')

ax3 = fig.add_subplot(224)
ax3.imshow(grady[:,0:797],cmap=colmap)
ax3.set_title('grady gpu')
ax3.axis('off')

plt.show()



fig2 = plt.figure(2)
plt.imshow(dist,cmap=colmap)

ax0 = fig2.add_subplot(221)
ax0.imshow(skex.rescale_intensity(gradxx, out_range=(0, 255)),cmap='Greys_r')
ax0.set_title('gradxx')
ax0.axis('off')

ax1 = fig2.add_subplot(222)
ax1.imshow(skex.rescale_intensity(gradxy, out_range=(0, 255)),cmap='Greys_r')
ax1.set_title('gradxy')
ax1.axis('off')

ax2 = fig2.add_subplot(223)
ax2.imshow(skex.rescale_intensity(gradyy, out_range=(0, 255)),cmap='Greys_r')
ax2.set_title('gradyy')
ax2.axis('off')

ax3 = fig2.add_subplot(224)
ax3.imshow(skex.rescale_intensity(div, out_range=(0, 255)),cmap='Greys_r')
ax3.set_title('anisotropic diffusion operator')

plt.show()

C = linalg.matAdd(image[:,:,0],image[:,:,1],0.5,0.5)


t1 = time.time()

    
cputime = time.time()-t1
print('cpu time'+str(cputime))

"""

#===========================================================================
# function evolution curve
#===========================================================================

#need to implement eikonal equation for signed distance function calculation away from a curve

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


final = 1*dist
#need to add space discretisation, i am getting blow up!
maxit    = 5000
timeStep = np.float32(0.001)


for n in range(0,maxit):
    final = mcm.onestepiteration(final,timeStep)
    final = final/(np.max(final)-np.min(final))
    #final = final + mcm.blockAnisDiffOperator(final)
    


fig2 = plt.figure(2)


ax0 = fig2.add_subplot(211)
ax0.imshow(dist,cmap='Greys_r')
ax0.set_title('initial condition')
ax0.axis('off')

ax1 = fig2.add_subplot(212)
ax1.imshow(final,cmap='Greys_r')
ax1.set_title('after t time steps')
ax1.axis('off')

#try cpu version