from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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

########################################################################
### Parallel rgb2gray transformation ###################################
########################################################################

image = scm.imread('546h2016_1_0.jpg').astype(np.float32)



a = image

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
plt.imshow(image,cmap='Greys_r')
plt.show()



########################################################################
### Parallel min max thresholding  #####################################
########################################################################

rasengan = 1

while rasengan != 0:
	#tempIm = np.empty_like (image)
	#np.copyto(tempIm, image) 
	tempIm = 1*image[:,:].reshape(aSize,order='F')
	image  = image.reshape(aSize,order='F')
	dest1  = tempIm
	
	# setting values of mu and sigma 
	minima = np.float32(input("enter value of min: "))
	print("min =", minima)
	maxima = np.float32(input("enter value of max: "))
	print("max =", maxima)
	
	
	# calling the gpu implementation of contrast enhancement
	mcm.GPUthresholding(drv.InOut(dest1),ydim,minima,maxima,block=(1024, 1, 1), grid=(1024, 1, 1))
	
	# reshaping the images
	dest1  = np.reshape(dest1,(a.shape[0],a.shape[1]), order='F')
	tempIm = np.reshape(tempIm,(a.shape[0],a.shape[1]), order='F')
	image  = np.reshape(image,(a.shape[0],a.shape[1]), order='F')
	
	print("doton: doryuuheki")
	
	# showing the enhanced image
	#plt.imshow(dest,cmap='Greys_r')
	#plt.show()
	fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
	ax0, ax1, ax2 = axes
	plt.gray()

	ax0.imshow(image)
	ax0.set_title('Original image')

	ax1.imshow(tempIm)
	ax1.set_title('TempIm')

	ax2.imshow(dest1)
	ax2.set_title('enhanced image')

	for ax in axes:
		ax.axis('off')

	plt.show()
	rasengan = np.int(input("exit? 0 for exit, 1 for no: "))



"""

########################################################################
### Parallel exp enhancement   ########################################
########################################################################

rasengan = 1

while rasengan != 0:
	#tempIm = np.empty_like (image)
	#np.copyto(tempIm, image) 
	tempIm = 1*image[:,:].reshape(aSize,order='F')
	image  = image.reshape(aSize,order='F')
	dest1  = tempIm
	
	# setting values of mu and sigma 
	mu = np.float32(input("enter value of mu: "))
	print("mu =", mu)
	sigma = np.float32(input("enter value of sigma: "))
	print("sigma =", sigma)
	
	
	# calling the gpu implementation of contrast enhancement
	expGPUenhancement(drv.Out(dest1), drv.In(tempIm),ydim,mu,sigma,block=(1024, 1, 1), grid=(1024, 1, 1))
	
	# reshaping the images
	dest1  = np.reshape(dest1,(a.shape[0],a.shape[1]), order='F')
	tempIm = np.reshape(tempIm,(a.shape[0],a.shape[1]), order='F')
	image  = np.reshape(image,(a.shape[0],a.shape[1]), order='F')
	
	print("doton: doryuuheki")
	
	# showing the enhanced image
	#plt.imshow(dest,cmap='Greys_r')
	#plt.show()
	fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
	ax0, ax1, ax2 = axes
	plt.gray()

	ax0.imshow(image)
	ax0.set_title('Original image')

	ax1.imshow(tempIm)
	ax1.set_title('TempIm')

	ax2.imshow(dest1)
	ax2.set_title('enhanced image')

	for ax in axes:
		ax.axis('off')

	plt.show()
	rasengan = np.int(input("exit? 0 for exit, 1 for no: "))



#=======================================================
#=======================================================






rasengan = 1
while rasengan != 0:
	global_thresh = threshold_otsu(image)
	binary_global = image > global_thresh

	block_size = int(input('block size for adaptive thresholding: '))
	binary_adaptive = threshold_adaptive(image, block_size, offset=10)

	fig, axes = plt.subplots(nrows=3, figsize=(10, 12))
	ax0, ax1, ax2 = axes
	plt.gray()

	ax0.imshow(image)
	ax0.set_title('Image')

	ax1.imshow(binary_global)
	ax1.set_title('Global thresholding')

	ax2.imshow(binary_adaptive)
	ax2.set_title('Adaptive thresholding')

	for ax in axes:
		ax.axis('off')

	plt.show()
	rasengan = np.int(input("exit? 0 for exit, 1 for no: "))

"""
