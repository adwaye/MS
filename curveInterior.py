import os
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


%matplotlib inline
image = scm.imread('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/fastmsToying/trialimages/curve3.jpg').astype(np.float32)
image = mcm.grayfication(image)
image[image.nonzero()]=1
plt.imshow(image,cmap='Greys_r')


def curveInterior(image):
    forme = image.shape
    if(np.size(forme)>2):
        sys.exit('Only works on gray images')
    

    aSize    = forme[0]*forme[1]
    xdim     = np.int32(forme[0])
    ydim     = np.int32(forme[1])   
    labIm    = np.zeros(forme)    
    for i in range(1,xdim):
        arr     = (1-image[i,:]).nonzero()
        larr    = len(arr)
        if(larr>1):
            trueArr = np.array((arr[0]))
            for k in range(1,larr):
                if(arr[k]-arr[k-1]>1):
                    trueArr =  np.append(trueArr,arr[k])
            N = len(trueArr)
            for l in range(0,N):
                if(l%2==0): labIm[i,:][trueArr[l]:trueArr[l+1]]=1
                print('we in there')
                    
        
    return labIm
"""    
    for j in range(1,ydim):
            if(labIm[i,j-1]!=0): 
                labIm[i,j]=image[i,j-1]
            else:
                if(image[i-1,j]!=0):
                    labIm[i,j]=image[i-1,j]
                else:
                    Nlab       = Nlab+1
                    labIm[i,j] = Nlab
    for j in range(0,ydim):
        arr     = (1-image[:,j]).nonzero()
        if(np.size(arr)!=0):        
            maxind  = np.max(arr)
            minind  = np.min(arr)
            labIm[:,j][minind:maxind] = 0
"""                        
    
%matplotlib inline
labIm                      = curveInterior(image)
plt.imshow(labIm)
#plt.hist(labIm.ravel())
