# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:11:16 2017

@author: amr62
"""

########################################################################
### Aim of this code: smooth the image with fastms, then use min-max ###
### thresholding to eliminate background of the picture to obtain    ###
### a clean image without patient names and extra bones              ### 
########################################################################


#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import scipy.misc as scm
import time

from skimage.filters import threshold_otsu, threshold_adaptive
#from myCudaModule import rgb2gray,expGPUenhancement,expGPUthresholding
import myCudaModule as mcm

import os

patientID = "51h2011"




########################################################################
### smoothing the image with the fastms code ###########################
########################################################################

#image = scm.imread('546h2016_1_0.jpg').astype(np.float32)
lambdastr = str(input("enter value of lambda: "))
order66  = "cd /home/amr62/fastms; ./main -edges 0 -i /home/amr62/Documents/TheEffingPhDHatersGonnaHate/fastmsToying/trialimages/"+patientID+"_1_0.jpg -alpha -1 -lambda "+ lambdastr+" -show 0 -save '/' >> /home/amr62/Documents/TheEffingPhDHatersGonnaHate/fastmsToying/trialimages/result.txt" 
os.system(order66) # this works
#os.system("sh order66.sh") # this works
#segmentedimage = scm.imread('546h2016_1_0__result_alphaInfinity_lambda0.6_edges.jpg').astype(np.float32)

imagetoloadStr = patientID+"_1_0__result_alphaInfinity_lambda"+lambdastr+".png"
segmentedIM    = scm.imread(imagetoloadStr).astype(np.float32)



########################################################################
### Parallel rgb2gray transformation ###################################
########################################################################



a = segmentedIM

aSize = a.shape[0]*a.shape[1]
r_img = a[:, :, 0].reshape(aSize, order='F')
g_img = a[:, :, 1].reshape(aSize, order='F')
b_img = a[:, :, 2].reshape(aSize, order='F')
dest=r_img
ydim   = np.int32(a.shape[1])

#finding size of grid
katon  = a.shape[0]*a.shape[1]/float(a.shape[1])
if katon - int(katon)>0:
	katon  = int(katon)+1
	print("remainder found")
else:
	katon = int(katon)
	print("exact division")

gridX  = katon
blockX =  a.shape[1]
#parallel rgb computation+time
t = time.time()
mcm.rgb2gray(drv.Out(dest), drv.In(r_img), drv.In(g_img),drv.In(b_img),ydim,block=(1024, 1, 1), grid=(1024, 1, 1))
ParTime = time.time()-t

image = np.reshape(dest,(a.shape[0],a.shape[1]), order='F')

print("Katon: Goukakyuu no jutsu")
print("parallel time=")
print(ParTime)
#plt.imshow(image,cmap='Greys_r')
#plt.show()



originalIM = scm.imread(patientID+"_1_0.jpg").astype(np.float32)




########################################################################
### Parallel min max thresholding  #####################################
########################################################################


a = originalIM

aSize = a.shape[0]*a.shape[1]
r_img = a[:, :, 0].reshape(aSize, order='F')
g_img = a[:, :, 1].reshape(aSize, order='F')
b_img = a[:, :, 2].reshape(aSize, order='F')
dest=r_img
ydim   = np.int32(a.shape[1])

#finding size of grid
katon  = a.shape[0]*a.shape[1]/float(a.shape[1])
if katon - int(katon)>0:
	katon  = int(katon)+1
	print("remainder found")
else:
	katon = int(katon)
	print("exact division")

gridX  = katon
blockX =  a.shape[1]
#parallel rgb computation+time
t = time.time()
mcm.rgb2gray(drv.Out(dest), drv.In(r_img), drv.In(g_img),drv.In(b_img),ydim,block=(1024, 1, 1), grid=(1024, 1, 1))
ParTime = time.time()-t

originalIM = np.reshape(dest,(a.shape[0],a.shape[1]), order='F')


nbin = np.int32(input('enter value of bin number: '))

fig = plt.figure(2)

ax0 = fig.add_subplot(221)
ax0.imshow(image,cmap='Greys_r')
ax0.set_title('segmented image with lambda='+lambdastr)
ax0.axis('off')

ax1 = fig.add_subplot(222)
ax1.hist(image,bins=nbin)
ax1.set_title('colour histogram of segmented image with lambda='+lambdastr)

ax2 = fig.add_subplot(223)
ax2.imshow(originalIM,cmap='Greys_r')
ax2.set_title('original image')
ax2.axis('off')

ax3 = fig.add_subplot(224)
ax3.hist(originalIM,bins=nbin)
ax3.set_title('colour histogram of original image')

plt.show()





rasengan = 1

while rasengan != 0:
	#tempIm = np.empty_like (image)
	#np.copyto(tempIm, image) 
 tempIm = 1*originalIM[:,:].reshape(aSize,order='F')
 image  = image.reshape(aSize,order='F')
 dest1  = tempIm
	
	# setting values of mu and sigma 
 minima = np.float32(input("enter value of min: "))
 print("min =", minima)
 maxima = np.float32(input("enter value of max: "))
 print("max =", maxima)
	
	
	# calling the gpu implementation of contrast enhancement
 mcm.GPUthresholding2(drv.InOut(dest1),drv.In(image),ydim,minima,maxima,block=(1024, 1, 1), grid=(1024, 1, 1))
	
	# reshaping the images
 dest1  = np.reshape(dest1,(a.shape[0],a.shape[1]), order='F')
 tempIm = np.reshape(tempIm,(a.shape[0],a.shape[1]), order='F')
 image  = np.reshape(image,(a.shape[0],a.shape[1]), order='F')
	
 print("doton: doryuuheki")
	
	# showing the enhanced image
	#plt.imshow(dest,cmap='Greys_r')
	#plt.show()
 fig = plt.figure(3)
     
 ax0 = fig.add_subplot(211)
 ax0.imshow(dest1,cmap='Greys_r')
 ax0.set_title(str(minima)+'<thresholded image<'+str(maxima))
 ax0.axis('off')     
     
 ax1 = fig.add_subplot(212)
 ax1.imshow(originalIM,cmap='Greys_r')
 ax1.set_title('Original Image')
 ax1.axis('off')
     
	#ax2.imshow(dest1)
	#ax2.set_title('enhanced image')

 plt.show()
 rasengan = np.int(input("exit? 0 for exit, 1 for no: "))
