#===================================================
# functions in this module:
# grayfication(image) : input colour image, output black and white
# threshold(image,minimum,maximum): input black and white image, min, max, output min<output(i,j)<max
# threshold2(image1,image2,minimum, maximum): input black and white image1, image2, outpuits image1 thesholded on the values of image2
# grad(image): input image, output gradx, grady, uses one Cuda thread for both gradient direction computations
# grad1(image): input image, output gradx, grady, uses one Cuda thread for each gradient direction computations
# secOrderGrad(image): input image, output gradxx, gradxy , gradyy, uses one Cuda thread gradxx and gradyy and a different one for gradxy
# anDiffOperator(image): input image, output is div, which is 
#  |\nabla f| \text{div} ( \frac{\nabla f}{|\nabla f|} )
# onestepIteration(image,timestep,maxit): evolves function image via the above with timestep=timestep and neumann boundary conditions
#===================================================


import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import sys


#===================================================
#=== parallel rgb2gray implementation ==============
#===================================================

mod = SourceModule \
    (
        """
#include<stdio.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)

__global__ void rgb2gray(float *dest,float *r_img, float *g_img, float *b_img, int ydim)
{

unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
  
  unsigned int yshape = ydim;
  unsigned int a      = idx/yshape;
  unsigned int b      = idx%yshape;
  
dest[INDEX(a, b, yshape)] = (0.299*r_img[INDEX(a, b, yshape)]+0.587*g_img[INDEX(a, b, yshape)]+0.114*b_img[INDEX(a, b, yshape)]);


}

"""
    )


rgb2gray = mod.get_function("rgb2gray")


#wrapper function that does the grayfication using only an image as input
def grayfication(image):
    forme = image.shape
    aSize = forme[0]*forme[1]
    xdim  = np.int32(forme[0])
    ydim  = np.int32(forme[1])
    r_img = image[:, :, 0].reshape(aSize, order='F')
    g_img = image[:, :, 1].reshape(aSize, order='F')
    b_img = image[:, :, 2].reshape(aSize, order='F')
    dest  = np.zeros(aSize).astype(np.float32)
    
    #block size: B := dim1*dim2*dim3=1024
    #gird size : dim1*dimr2*dim3 = ceiling(aSize/B)
    blockX     = int(xdim)
    multiplier = aSize/float(blockX)   
    if(aSize/float(blockX) > int(aSize/float(blockX)) ):
        gridX = int(multiplier + 1)
    else:
        gridX = int(multiplier)
        

#parallel rgb computation+time
    rgb2gray(drv.Out(dest), drv.InOut(r_img),      drv.InOut(g_img),drv.InOut(b_img),ydim,block=(blockX, 1, 1), grid=(gridX, 1, 1))

    dest = np.reshape(dest,forme[0:2], order='F')
    return dest

#====================================================
#=== parallel contrast enhancement using exp ========
#====================================================

sasuke	 = SourceModule \
    (
        """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)

__global__ void expGPUenhancement(float *dest,float *img, int ydim, float mu, float sigma)
{

unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
  
  unsigned int yshape = ydim;
  unsigned int a      = idx/yshape;
  unsigned int b      = idx%yshape;
  
dest[INDEX(a, b, yshape)] = exp((img[INDEX(a, b, yshape)]-mu)/sigma);


}


"""
    )


expGPUenhancement = sasuke.get_function("expGPUenhancement") 



#====================================================
#=== parallel thresholding ==========================
#====================================================

uchiha	 = SourceModule \
    (
        """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)

__global__ void GPUthresholding(float *img, int ydim, float min, float max)
{

unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
  
  unsigned int yshape = ydim;
  unsigned int a      = idx/yshape;
  unsigned int b      = idx%yshape;
  
if(img[INDEX(a, b, yshape)] > max || img[INDEX(a, b, yshape)] < min) {
   img[INDEX(a, b, yshape)] = 0.0;
}
/*else {
   
}
*/

}

"""
    )

GPUthresholding = uchiha.get_function("GPUthresholding")


#wrapper function that does threshold image values to be in minimum<x< maximum
def threshold(image,minimum,maximum):    
    forme = image.shape
    if(np.size(forme)>2):
        sys.exit('Only works on gray images')
        
        
    aSize = forme[0]*forme[1]
    xdim  = np.int32(forme[0])
    ydim  = np.int32(forme[1])
    dest  = np.zeros(aSize).astype(np.float32)
    
    
    minval = np.float32(minimum)
    maxval = np.float32(maximum)
    
    #block size: B := dim1*dim2*dim3=1024
    #gird size : dim1*dimr2*dim3 = ceiling(aSize/B)
    blockX     = int(1024)
    multiplier = aSize/float(1024)   
    if(aSize/float(1024) > int(aSize/float(1024)) ):
        gridX = int(multiplier + 1)
    else:
        gridX = int(multiplier)
#parallel rgb computation+time
    GPUthresholding(drv.InOut(dest), ydim, minval, maxval,block=(blockX, 1, 1), grid=(gridX, 1, 1))

    dest = np.reshape(dest,forme[0:2], order='F')
    return dest
    

#====================================================
#=== parallel thresholding using values =============
#=== of segmented image                 =============      
#=== thresholds image *img using values =============
#=== of segmented image *segIm
#====================================================

uzumaki = SourceModule \
    (
        """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)

__global__ void GPUthresholding2(float *img, float *segIm, int ydim, float min, float max)
{

unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
  
  unsigned int yshape = ydim;
  unsigned int a      = idx/yshape;
  unsigned int b      = idx%yshape;
  
if(segIm[INDEX(a, b, yshape)] > max || segIm[INDEX(a, b, yshape)] < min) {
   img[INDEX(a, b, yshape)] = 0.0;
}
/*else {
   
}
*/

}

"""
    )

GPUthresholding2 = uzumaki.get_function("GPUthresholding2")


# wrapper function that does the thresholding on image image1 based on the values of image2 (usually a smoothed, segmented version of image1)  to lie in (minimum,maximum)
def threshold2(image1,image2,minimum, maximum):
    forme1 = image1.shape
    forme2 = image2.shape
    if(np.size(forme1)>2 & np.size(forme2)>2):
        sys.exit('Only works on gray images')
        
        
    aSize  = forme1[0]*forme1[1]
    xdim   = np.int32(forme1[0])
    ydim   = np.int32(forme1[1])
    dest   = np.zeros(aSize).astype(np.float32)
    image2 = image2.reshape(aSize, order='F')
    
    minval = np.float32(minimum)
    maxval = np.float32(maximum)
    
    #block size: B := dim1*dim2*dim3=1024
    #gird size : dim1*dimr2*dim3 = ceiling(aSize/B)
    blockX     = int(1024)
    multiplier = aSize/float(1024)   
    if(aSize/float(1024) > int(aSize/float(1024)) ):
        gridX = int(multiplier + 1)
    else:
        gridX = int(multiplier)
#parallel rgb computation+time
    GPUthresholding2(drv.InOut(dest),drv.In(image2),ydim,minima,maxima,block=(blockX, 1, 1), grid=(gridX, 1, 1))

    dest = np.reshape(dest,forme1[0:2], order='F')
    return dest
    


#===========================================================================
#== parallel gradient computations
#== input arrays are shleft : shleft(i,j) = image(i-1,j)
#==                  shright: shleft(i,j) = image(i+1,j)
#==                  shup   : shup(i,j)   = image(i,j+1)
#==                  shdown : shdown(i,j) = image(i,j-1)
#== output arrays    gradx(i,j) = (image(i+1,j)-image(i-1,j))/2     
#==                  grady(i,j) = (image(i,j+1)-image(i,j-1))/2
#============================================================================


#sourcemodule for x-y gradient calculation in parallel
marcusKaiser = SourceModule \
    ( 
"""
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)

__global__ void GradientCalculation1(float *gradx, float *grady, float *image,  int ydim, int xdim)
{

  unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a   = idx/ydim;
  unsigned int b   = idx%ydim;
  unsigned int xlim = xdim-1;
  unsigned int ylim = ydim-1;
  

  
 if( b< ylim && b>0){
   gradx[INDEX(a, b, ydim)] = 0.5*(image[INDEX(a, b+1, ydim)]-image[INDEX(a, b-1, ydim)]);   
 
 } 


 if(a<xlim && a >0){
  grady[INDEX(a, b, ydim)] = 0.5*( image[INDEX(a+1, b, ydim)]-image[INDEX(a-1, b, ydim)]);
 }
 

} 
"""
    )


GradientCalculation1 = marcusKaiser.get_function("GradientCalculation1")

#wrapper for gradient calculation of both x and y directions
# outputs gradx, grady, which are gradients in x, y direction respectively
def grad(image):
    forme = image.shape
    if(np.size(forme)>2):
        sys.exit('Only works on gray images')

    aSize = forme[0]*forme[1]
    xdim  = np.int32(forme[0])
    ydim  = np.int32(forme[1])    
    
    #setiing up the shifted image matrices
    
    
    #setting the gradient matrices
    gradx = np.zeros(aSize).astype(np.float32)
    grady = np.zeros(aSize).astype(np.float32)
    
    #reshaping the image matrix
    image = image.reshape(aSize,order= 'F')
    #dist   = dist.reshape(aSize,order='F')
    #block size: B := dim1*dim2*dim3=1024
    #gird size : dim1*dimr2*dim3 = ceiling(aSize/B)
    blockX     = int(1024)
    multiplier = aSize/float(blockX)   
    if(aSize/float(blockX) > int(aSize/float(blockX)) ):
        gridX = int(multiplier + 1)
    else:
        gridX = int(multiplier)
    
    #launching the cthread
    GradientCalculation1(drv.Out(gradx), drv.Out(grady),drv.In(image), ydim, xdim, block=(blockX, 1, 1),grid=(gridX, 1, 1))
    
    # reshaping the output as a 2d array
    gradx = np.reshape(gradx, forme[0:2], order='F')
    grady = np.reshape(grady, forme[0:2], order='F')
    return gradx, grady


#sourcemodule for only x gradient calculation
benRobinson = SourceModule \
    (
        """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)

__global__ void GradientCalculationY(float *grady, float *shleft, float *shright, int ydim, int xdim) 
{
  unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a      = idx/ydim;
  unsigned int b      = idx%ydim;
  
if( a< xdim-1 && a > 0 && b < ydim-1 && b > 0){
  grady[INDEX(a, b, ydim)] = 0.5*(shright[INDEX(a, b, ydim)]-shleft[INDEX(a, b, ydim)]);

} else {
  grady[INDEX(a, b, ydim)] = 0.0;  
}  
  
}          
"""    
    )



GradientCalculationY = benRobinson.get_function("GradientCalculationY")



#sourcemodule for only x gradient calculation




groKok = SourceModule \
    (
        """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)

__global__ void GradientCalculationX(float *gradx, float *shup, float *shdown, int ydim, int xdim) 
{
  unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a      = idx/ydim;
  unsigned int b      = idx%ydim;
  
if( a< xdim-1 && a > 0 && b < ydim-1 && b > 0){
  gradx[INDEX(a, b, ydim)] = 0.5*(shup[INDEX(a, b, ydim)]-shdown[INDEX(a, b, ydim)]);

} else {
  gradx[INDEX(a, b, ydim)] = 0.0;  
}  
  
}        
    
"""    
    )



GradientCalculationX = groKok.get_function("GradientCalculationX")


#wrapper for gradient calculation, code uses separate threads for y direction and x direction calculations
# outputs gradx, grady, which are gradients in x, y direction respectively

def grad1(image):
    forme = image.shape
    if(np.size(forme)>2):
        sys.exit('Only works on gray images')

    aSize = forme[0]*forme[1]
    xdim  = np.int32(forme[0])
    ydim  = np.int32(forme[1])    
    
    #setiing up the shifted image matrices
    shleft  = np.zeros(forme[0:2]).astype(np.float32)
    shright = np.zeros(forme[0:2]).astype(np.float32)
    shup    = np.zeros(forme[0:2]).astype(np.float32)
    shdown  = np.zeros(forme[0:2]).astype(np.float32)
    
    shleft[1:(xdim-1),:]  = image[0:(xdim-2),:] #shleft: shleft(i,j) = image(i-1,j)
    shleft                = shleft.reshape(aSize,order= 'F')
    
    shright[0:(xdim-2),:] = image[1:(xdim-1),:] #shright: shleft(i,j) = image(i+1,j)
    shright               = shright.reshape(aSize,order= 'F')    
    
    shup[:,0:(ydim-2)]    = image[:,1:(ydim-1)] #shup  : shup(i,j)   = image(i,j+1)
    shup                  = shup.reshape(aSize,order= 'F')
    
    shdown[:,1:(ydim-1)]  = image[:,0:(ydim-2)] #shdown : shdown(i,j) = image(i,j-1)
    shdown                = shdown.reshape(aSize,order= 'F')
    
    #setting the gradient matrices
    gradx = np.zeros(aSize).astype(np.float32)
    grady = np.zeros(aSize).astype(np.float32)
    
    #reshaping the image matrix
    image = image.reshape(aSize,order= 'F')
    
    #block size: B := dim1*dim2*dim3=1024
    #gird size : dim1*dimr2*dim3 = ceiling(aSize/B)
    blockX     = int(1024)
    multiplier = aSize/float(1024)   
    if(aSize/float(1024) > int(aSize/float(1024)) ):
        gridX = int(multiplier + 1)
    else:
        gridX = int(multiplier)
    
    #launching the cthread
    GradientCalculationX(drv.Out(gradx),drv.In(shleft), drv.In(shright), ydim, xdim, block=(blockX, 1, 1),grid=(gridX, 1, 1))
    GradientCalculationY(drv.Out(grady), drv.In(shup), drv.In(shdown), ydim, xdim, block=(blockX, 1, 1),grid=(gridX, 1, 1))    
    
    # reshaping the output as a 2d array
    gradx = np.reshape(gradx, forme[0:2], order='F')
    grady = np.reshape(grady, forme[0:2], order='F')
    return gradx, grady



    
    
    
#===========================================================================
#== parallel second order derivatives computations
#== input arrays are image        : function array
#==                  shleft       : shleft(i,j)      = image(i-1,j)
#==                  shright      : shleft(i,j)      = image(i+1,j)
#==                  shup         : shup(i,j)        = image(i,j+1)
#==                  shdown       : shdown(i,j)      = image(i,j-1)
#==                  shdupright   : shdupright(i,j)  = image(i+1,j+1)
#==                  shdupleft    : shdupleft(i,j)   = image(i-1,j+1)
#==                  shddownright : shdownright(i,j) = image(i+1,j-1)
#==                  shdownleft   : shdownleft(i,j)  = image(i-1,j-1)    
#==                  
#== tikok is the sourcemodule that contains the xx, yy direction computation
#== gradxx, gradxy, gradyy
#============================================================================
    
    
# calculates only gradxx, gradyy    
tikok = SourceModule \
    (
    """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)
    
__global__ void partialXXYYDerivCalculation(float* gradxx,float *gradyy, float *image, float *shleft, float *shright,float *shup, float *shdown, int ydim, int xdim)
{
  unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a      = idx/ydim;
  unsigned int b      = idx%ydim;
  
  if( a< xdim-1 && a > 0 && b < ydim-1 && b > 0){
  gradxx[INDEX(a, b, ydim)] = shright[INDEX(a, b, ydim)]- 2*image[INDEX(a, b, ydim)] +  shleft[INDEX(a, b, ydim)];
  gradyy[INDEX(a, b, ydim)] = shup[INDEX(a, b, ydim)]-2*image[INDEX(a, b, ydim)] + shdown[INDEX(a, b, ydim)];  
} else {
  gradxx[INDEX(a, b, ydim)] = 0.0;
  gradyy[INDEX(a, b, ydim)] = 0.0;  
}  

} 
"""    
    )
    
partialXXYYDerivCalculation = tikok.get_function("partialXXYYDerivCalculation")

    
# calculates gradxy only



# calculates only gradxx, gradyy    
moyenkok = SourceModule \
    (
    """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)
    
__global__ void partialXYDerivCalculation(float *gradxy, float *shupright, float *shupleft , float *shdownright, float *shdownleft, int ydim, int xdim)
{

  unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a      = idx/ydim;
  unsigned int b      = idx%ydim;
  
  if( a< xdim-1 && a > 0 && b < ydim-1 && b > 0){
  gradxy[INDEX(a, b, ydim)] = 0.25*(shupright[INDEX(a, b, ydim)]-  shupleft[INDEX(a,b,ydim)]- shdownright[INDEX(a, b, ydim)] + shdownleft[INDEX(a, b, ydim)]);
  } else {
  gradxy[INDEX(a, b, ydim)] = 0.0;
}  

}     
"""   
    )
    
    
    
partialXYDerivCalculation = moyenkok.get_function("partialXYDerivCalculation")

# wrapper for partial derivative calculations
# inputs black and white image
# outpus gradxx, gradxy, gradyy
def secOrderGrad(image):
    forme = image.shape
    if(np.size(forme)>2):
        sys.exit('Only works on gray images')

    aSize = forme[0]*forme[1]
    xdim  = np.int32(forme[0])
    ydim  = np.int32(forme[1])    
    
    #setiing up the shifted image matrices
    shleft  = np.zeros(forme[0:2]).astype(np.float32)
    shright = np.zeros(forme[0:2]).astype(np.float32)
    shup    = np.zeros(forme[0:2]).astype(np.float32)
    shdown  = np.zeros(forme[0:2]).astype(np.float32)
    
    shleft[1:(xdim-1),:]  = image[0:(xdim-2),:] #shleft: shleft(i,j) = image(i-1,j)
    shleft                = shleft.reshape(aSize,order= 'F')
    
    shright[0:(xdim-2),:] = image[1:(xdim-1),:] #shright: shleft(i,j) = image(i+1,j)
    shright               = shright.reshape(aSize,order= 'F')    
    
    shup[:,0:(ydim-2)]    = image[:,1:(ydim-1)] #shup  : shup(i,j)   = image(i,j+1)
    shup                  = shup.reshape(aSize,order= 'F')
    
    shdown[:,1:(ydim-1)]  = image[:,0:(ydim-2)] #shdown : shdown(i,j) = image(i,j-1)
    shdown                = shdown.reshape(aSize,order= 'F')
    
    #setting the gradient matrices
    gradxx = np.zeros(aSize).astype(np.float32)
    gradyy = np.zeros(aSize).astype(np.float32)
    gradxy = np.zeros(aSize).astype(np.float32)
    
    #reshaping the image matrix
    image = image.reshape(aSize,order= 'F')
    
    #block size: B := dim1*dim2*dim3=1024
    #gird size : dim1*dimr2*dim3 = ceiling(aSize/B)
    blockX     = int(1024)
    multiplier = aSize/float(1024)   
    if(aSize/float(1024) > int(aSize/float(1024)) ):
        gridX = int(multiplier + 1)
    else:
        gridX = int(multiplier)
    
    #cuda thread that does the calculation XX, yy calculation
    partialXXYYDerivCalculation(drv.Out(gradxx),drv.Out(gradyy), drv.In(image), drv.In(shleft), drv.In(shright),drv.In(shup), drv.In(shdown), ydim,  xdim, block = (blockX, 1, 1),grid=(gridX, 1, 1))
    """
    del shleft,shright,shup,shdown
    
        #setiing up the shifted image matrices
    shupleft    = np.zeros(forme[0:2]).astype(np.float32)
    shupright   = np.zeros(forme[0:2]).astype(np.float32)
    shdownright = np.zeros(forme[0:2]).astype(np.float32)
    shdownleft  = np.zeros(forme[0:2]).astype(np.float32)
    """
    #reshaping arrays
    shup    = np.reshape(shup, forme[0:2], order='F')
    shleft  = np.reshape(shleft, forme[0:2], order='F')
    shdown  = np.reshape(shdown, forme[0:2], order='F')
    shright = np.reshape(shright, forme[0:2], order='F')
    image   = np.reshape(image, forme[0:2], order='F')
    
    shup[0:(xdim-2),0:(ydim-2)]    = image[1:(xdim-1),1:(ydim-1)] #shdupright(i,j)  = image(i+1,j+1)
    shup                           = shup.reshape(aSize,order= 'F')
    
    shleft[1:(xdim-1),0:(ydim-2)]  = image[0:(xdim-2),1:(ydim-1)] # shupleft(i,j)   = image(i-1,j+1)
    shleft                         = shleft.reshape(aSize,order= 'F')    
    
    shdown[0:(xdim-2),1:(ydim-1)]  = image[1:(xdim-1),0:(ydim-2)] #shdownright(i,j) = image(i+1,j-1)
    shdown                         = shdown.reshape(aSize,order= 'F')
    
    shright[1:(xdim-1),1:(ydim-1)] = image[0:(xdim-2),0:(ydim-2)] # shdownleft(i,j)  = image(i-1,j-1)   
    shright                        = shright.reshape(aSize,order= 'F')
    
    image = image.reshape(aSize,order='F')

#partialXYDerivCalculation(gradxy, shupright, shupleft , shdownright, shdownleft, ydim, xdim)
    partialXYDerivCalculation(drv.Out(gradxy), drv.In(shup), drv.In(shleft) , drv.In(shdown), drv.In(shright), ydim, xdim, block = (blockX, 1, 1),grid=(gridX, 1, 1))
    
    gradxx = np.reshape(gradxx, forme[0:2], order='F')
    gradxy = np.reshape(gradxy, forme[0:2], order='F')
    gradyy = np.reshape(gradyy, forme[0:2], order='F')    
    return gradxx, gradxy, gradyy
    
   

#===========================================================================
#== parallel computation of anisotropic diffusion operator for a function
#== having a 2-d domain
#== |\nabla f| \text{div} ( \frac{\nabla f}{|\nabla f|}    )
#== input arrays are gradx, grady, gradxx, gradyy, gradxy    
#==                  
#== mujonameinu is the sourcemodule that contains the xx, yy direction computation
#== input  : gradx, grady, gradxx, gradxy, gradyy
#== output : gradx
#============================================================================

mujonaMeinu = SourceModule \
    (
    """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)
    
__global__ void anisotropicDiffusionOperator(float *gradx ,float *grady ,float *gradxx ,float *gradxy,float *gradyy,int ydim,int xdim)
{

  unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a      = idx/ydim;
  unsigned int b      = idx%ydim;
  if((gradx[INDEX(a,b,ydim)]*gradx[INDEX(a,b,ydim)]) + (grady[INDEX(a,b,ydim)]*grady[INDEX(a,b,ydim)]) > 0.001){
      gradx[INDEX(a,b,ydim)] = (  (gradxx[INDEX(a,b,ydim)]*grady[INDEX(a,b,ydim)]*grady[INDEX(a,b,ydim)]) - (2*gradx[INDEX(a,b,ydim)]*grady[INDEX(a,b,ydim)]*gradxy[INDEX(a,b,ydim)]) + (gradyy[INDEX(a,b,ydim)]*gradx[INDEX(a,b,ydim)]*gradx[INDEX(a,b,ydim)] ) )/( (gradx[INDEX(a,b,ydim)]*gradx[INDEX(a,b,ydim)]) + (grady[INDEX(a,b,ydim)]*grady[INDEX(a,b,ydim)]) );
  } else {
    gradx[INDEX(a,b,ydim)] = 0.0;  
  }
    

} 
    
    
"""    
    )
    
anisotropicDiffusionOperator = mujonaMeinu.get_function("anisotropicDiffusionOperator")
   
def anDiffOperator(image):
    forme = image.shape
    if(np.size(forme)>2):
        sys.exit('Only works on gray images')

    aSize = forme[0]*forme[1]
    xdim  = np.int32(forme[0])
    ydim  = np.int32(forme[1])    
    
    
    gradx, grady           = grad(image)
    gradxx, gradxy, gradyy = secOrderGrad(image)
        
    gradx  = gradx.reshape(aSize,order='F')
    grady  = grady.reshape(aSize,order='F')
    gradxx = gradxx.reshape(aSize,order='F')
    gradxy = gradxy.reshape(aSize,order='F')
    gradyy = gradx.reshape(aSize,order='F')
    
    #block size: B := dim1*dim2*dim3=1024
    #gird size : dim1*dimr2*dim3 = ceiling(aSize/B)
    blockX     = int(1024)
    multiplier = aSize/float(1024)   
    if(aSize/float(1024) > int(aSize/float(1024)) ):
        gridX = int(multiplier + 1)
    else:
        gridX = int(multiplier)
    
    #remember, in this thread, the output is stored in the first input, here gradx
    anisotropicDiffusionOperator(drv.InOut(gradx) ,drv.In(grady) ,drv.In(gradxx) ,drv.In(gradxy),drv.In(gradyy),ydim,xdim,block=(blockX,1,1),grid=(gridX,1,1))
    
    gradx = np.reshape(gradx,forme[0:2],order='F')
    
    return gradx








#===========================================================================
#== parallel computation of anisotropic diffusion operator for a function
#== having a 2-d domain
#== |\nabla f| \text{div} ( \frac{\nabla f}{|\nabla f|}    )
#== input arrays are gradx, grady, gradxx, gradyy, gradxy    
#== calculates the gradient, and 2nd order operations from scratch                 
#== 
#== saiyan is the sourcemodule that contains the xx, yy direction computation
#== input arrays are image        : function array
#==                  shleft       : shleft(i,j)      = image(i-1,j)
#==                  shright      : shleft(i,j)      = image(i+1,j)
#==                  shup         : shup(i,j)        = image(i,j+1)
#==                  shdown       : shdown(i,j)      = image(i,j-1)
#==                  shdupright   : shdupright(i,j)  = image(i+1,j+1)
#==                  shdupleft    : shdupleft(i,j)   = image(i-1,j+1)
#==                  shddownright : shdownright(i,j) = image(i+1,j-1)
#==                  shdownleft   : shdownleft(i,j)  = image(i-1,j-1)    
#== output : shleft
#============================================================================


    

saiyan = SourceModule \
    (
    """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)
    
__global__ void directGPUdiv(float *image,float *shleft, float *shright, float *shup, float *shdown,float *shupright, float *shupleft, float *shdownright, float *shdownleft,int ydim, int xdim)
{
  float gradx, grady, gradxx, gradxy, gradyy;
  
  unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a      = idx/ydim;
  unsigned int b      = idx%ydim;
  
  
  if( a< xdim-1 && a > 0 && b < ydim-1 && b > 0){
    gradxx = shright[INDEX(a, b, ydim)]- 2*image[INDEX(a, b, ydim)] +  shleft[INDEX(a, b, ydim)];
    gradyy = shup[INDEX(a, b, ydim)]-2*image[INDEX(a, b, ydim)] + shdown[INDEX(a, b, ydim)];  
    gradxy = 0.25*(shupright[INDEX(a, b, ydim)]-  shupleft[INDEX(a,b,ydim)]- shdownright[INDEX(a, b, ydim)] + shdownleft[INDEX(a, b, ydim)]);
    gradx = 0.5*(shright[INDEX(a, b, ydim)]-shleft[INDEX(a, b, ydim)]);
    grady = 0.5*(shup[INDEX(a, b, ydim)]-shdown[INDEX(a, b, ydim)]);  
} else {
    gradxx = 0.0;
    gradyy = 0.0;  
    gradxy = 0.0;
    gradx  = 0.0;
    grady  = 0.0;
}  

  if((gradx*gradx) + (grady*grady) > 0.001){
    shleft[INDEX(a,b,ydim)] = (  (gradxx*grady*grady) - (2*gradx*grady*gradxy) + (gradyy*gradx*gradx ) )/( (gradx*gradx) + (grady*grady) );
  } else {
    shleft[INDEX(a,b,ydim)] = 0.0;  
  }

} 
"""    
    )
    
directGPUdiv = saiyan.get_function("directGPUdiv")

#having shleft overwritten might cause stability issues    
#wrapper to calculate the anisotropic diffusion operator input is the image, output is an array with the value of 
#== |\nabla f| \text{div} ( \frac{\nabla f}{|\nabla f|}    )
        
def anisotropicDiffOperator(image):
    forme = image.shape
    if(np.size(forme)>2):
        sys.exit('Only works on gray images')

    aSize = forme[0]*forme[1]
    xdim  = np.int32(forme[0])
    ydim  = np.int32(forme[1])    
    
    #setiing up the shifted image matrices
    shleft  = np.zeros(forme[0:2]).astype(np.float32)
    shright = np.zeros(forme[0:2]).astype(np.float32)
    shup    = np.zeros(forme[0:2]).astype(np.float32)
    shdown  = np.zeros(forme[0:2]).astype(np.float32)
    
     
    shupleft    = np.zeros(forme[0:2]).astype(np.float32)
    shupright   = np.zeros(forme[0:2]).astype(np.float32)  
    shdownleft  = np.zeros(forme[0:2]).astype(np.float32)
    shdownright = np.zeros(forme[0:2]).astype(np.float32)
    
    
    shleft[1:(xdim-1),:]  = image[0:(xdim-2),:] #shleft: shleft(i,j) = image(i-1,j)
    shleft                = shleft.reshape(aSize,order= 'F')
    
    shright[0:(xdim-2),:] = image[1:(xdim-1),:] #shright: shleft(i,j) = image(i+1,j)
    shright               = shright.reshape(aSize,order= 'F')    
    
    shup[:,0:(ydim-2)]    = image[:,1:(ydim-1)] #shup  : shup(i,j)   = image(i,j+1)
    shup                  = shup.reshape(aSize,order= 'F')
    
    shdown[:,1:(ydim-1)]  = image[:,0:(ydim-2)] #shdown : shdown(i,j) = image(i,j-1)
    shdown                = shdown.reshape(aSize,order= 'F')
    
    shupright[0:(xdim-2),0:(ydim-2)]    = image[1:(xdim-1),1:(ydim-1)] #shdupright(i,j)  = image(i+1,j+1)
    shupright                           = shupright.reshape(aSize,order= 'F')
    
    shupleft[1:(xdim-1),0:(ydim-2)]  = image[0:(xdim-2),1:(ydim-1)] # shupleft(i,j)   = image(i-1,j+1)
    shupleft                         = shupleft.reshape(aSize,order= 'F')    
    
    shdownright[0:(xdim-2),1:(ydim-1)]  = image[1:(xdim-1),0:(ydim-2)] #shdownright(i,j) = image(i+1,j-1)
    shdownright                         = shdownright.reshape(aSize,order= 'F')
    
    shdownleft[1:(xdim-1),1:(ydim-1)] = image[0:(xdim-2),0:(ydim-2)] # shdownleft(i,j)  = image(i-1,j-1)   
    shdownleft                        = shdownleft.reshape(aSize,order= 'F')
        
    
    #reshaping the image matrix
    image = image.reshape(aSize,order= 'F')
    
    #block size: B := dim1*dim2*dim3=1024
    #gird size : dim1*dimr2*dim3 = ceiling(aSize/B)
    blockX     = int(1024)
    multiplier = aSize/float(1024)   
    if(aSize/float(1024) > int(aSize/float(1024)) ):
        gridX = int(multiplier + 1)
    else:
        gridX = int(multiplier)
    
    #cuda thread that does the calculation XX, yy calculation
    directGPUdiv(drv.In(image),drv.InOut(shleft), drv.In(shright), drv.In(shup), drv.In(shdown),drv.In(shupright), drv.In(shupleft), drv.In(shdownright), drv.In(shdownleft),ydim, xdim, block=(blockX,1,1), grid=(gridX,1,1))
        
    shleft = shleft.reshape(aSize,order='F')

    return shleft


  
    
    


#===========================================================================
#== parallel evolution of anisotropic diffusion operator for a function
#== having a 2-d domain with neumann boundary conditions
#== |\nabla f| \text{div} ( \frac{\nabla f}{|\nabla f|}    )
#== 
#== 
#== 
#== honoikazuchi is the sourcemodule that contains the 1 step evolution calcultaion on a gpu
#== input arrays are image : function array
#==                  
#== calculates the gradients, and 2nd order operations on the gpu without shifted matrices
#== output : final
#== input  : image, state at previous time point
#============================================================================


honoikazuchi = SourceModule \
    (
    """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)
    
__global__ void diffIteration(float *image,float *final,int ydim, int xdim, float timestep)
{
  float gradx, grady, gradxx, gradxy, gradyy;
  
  unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a      = idx/ydim;
  unsigned int b      = idx%ydim;
  
  
  if( a< xdim-1 && a > 0 && b < ydim-1 && b > 0){
    gradxx = image[INDEX(a, b, ydim)+1]- 2*image[INDEX(a, b, ydim)] +  image[INDEX(a, b, ydim)-1];
    gradyy = image[INDEX(a, b, ydim)+ydim]-2*image[INDEX(a, b, ydim)] + image[INDEX(a, b, ydim)-ydim];  
    gradxy = 0.25*(image[INDEX(a, b, ydim)+ydim+1] -image[INDEX(a, b, ydim)+ydim-1] -image[INDEX(a, b, ydim)-ydim+1] + image[INDEX(a, b, ydim)-ydim-1]);
    gradx  = 0.5*(image[INDEX(a, b, ydim)+1]-image[INDEX(a, b, ydim)-1]);
    grady  = 0.5*(image[INDEX(a, b, ydim)+ydim]-image[INDEX(a, b, ydim)-ydim]);
    if((gradx*gradx) + (grady*grady) > 0.0000001){
       final[INDEX(a,b,ydim)] = image[INDEX(a,b,ydim)] + timestep*(gradxx*grady*grady - 2*gradx*grady*gradxy + gradyy*gradx*gradx)/((gradx*gradx) + (grady*grady));
    } else {
       final[INDEX(a,b,ydim)] = image[INDEX(a,b,ydim)];
    }
} else {
    gradx  = 0.0;
    grady  = 0.0;
    gradxx = 0.0;
    gradxy = 0.0;
    gradyy = 0.0;
}




} 
"""    
    )

diffIteration = honoikazuchi.get_function("diffIteration")


def onestepIteration(dist,timestep,maxit):
    """
    iterates the function image on a 2d grid through an euler anisotropic
    diffusion operator with timestep=timestep maxit number of times
    """
    image = 1*dist
    forme = image.shape
    if(np.size(forme)>2):
        sys.exit('Only works on gray images')

    aSize = forme[0]*forme[1]
    xdim  = np.int32(forme[0])
    ydim  = np.int32(forme[1])  
    
    
    image[0,:]      = image[1,:]
    image[xdim-1,:] = image[xdim-2,:]
    image[:,ydim-1] = image[:,ydim-2]
    image[:,0]      = image[:,1]
     
    image = image.reshape(aSize,order= 'C').astype(np.float32)
    final = np.zeros(aSize).astype(np.float32)
    
    #reshaping the image matrix
    
    
    #block size: B := dim1*dim2*dim3=1024
    #gird size : dim1*dimr2*dim3 = ceiling(aSize/B)
    blockX     = int(1024)
    multiplier = aSize/float(1024)   
    if(aSize/float(1024) > int(aSize/float(1024)) ):
        gridX = int(multiplier + 1)
    else:
        gridX = int(multiplier)
    
    for k in range(0,maxit):
       diffIteration(drv.In(image),drv.Out(final),ydim, xdim, np.float32(timestep),block=(blockX,1,1), grid=(gridX,1,1))
       final = final.reshape(forme,order='C')
       final[0,:]      = final[1,:]
       final[xdim-1,:] = final[xdim-2,:]
       final[:,ydim-1] = final[:,ydim-2]
       final[:,0]      = final[:,1]  
       image           = final.reshape(aSize,order= 'C').astype(np.float32)

    return final.reshape(forme,order='C')






#===========================================================================
#== parallel evolution of anisotropic diffusion operator for a function
#== having a 2-d domain with neumann boundary conditions
#== |\nabla f| \text{div} ( \frac{\nabla f}{|\nabla f|}    )
#== 
#== This one does all the iterations on the effing GPU, #yolo #swaglife
#== #potatoesGonnaPotate #keepMirin
#== 
#== honoikazuchi is the sourcemodule that contains the 1 step evolution calcultaion on a gpu
#== input arrays are image : function array
#==                  
#== calculates the gradients, and 2nd order operations on the gpu without shifted matrices
#== output : final
#== input  : image, state at previous time point
#============================================================================



soussMoKok = SourceModule \
    (
    """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)
    
__global__ void diffIteration(float *image,float *final,int ydim, int xdim, float timestep,int maxit)
{
  float gradx, grady, gradxx, gradxy, gradyy;
  
  unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a = idx/ydim;
  unsigned int b = idx%ydim;
  unsigned int k =0;
  
  for(k=0;k < maxit;k=k+1){
  if( a< xdim-1 && a > 0 && b < ydim-1 && b > 0){
    gradxx = image[INDEX(a, b, ydim)+1]- 2*image[INDEX(a, b, ydim)] +  image[INDEX(a, b, ydim)-1];
    gradyy = image[INDEX(a, b, ydim)+ydim]-2*image[INDEX(a, b, ydim)] + image[INDEX(a, b, ydim)-ydim];  
    gradxy = 0.25*(image[INDEX(a, b, ydim)+ydim+1] -image[INDEX(a, b, ydim)+ydim-1] -image[INDEX(a, b, ydim)-ydim+1] + image[INDEX(a, b, ydim)-ydim-1]);
    gradx  = 0.5*(image[INDEX(a, b, ydim)+1]-image[INDEX(a, b, ydim)-1]);
    grady  = 0.5*(image[INDEX(a, b, ydim)+ydim]-image[INDEX(a, b, ydim)-ydim]);
    if((gradx*gradx) + (grady*grady) > 0.0000001){
       final[INDEX(a,b,ydim)] = image[INDEX(a,b,ydim)] + timestep*(gradxx*grady*grady - 2*gradx*grady*gradxy + gradyy*gradx*gradx)/((gradx*gradx) + (grady*grady));
    } else {
       final[INDEX(a,b,ydim)] = image[INDEX(a,b,ydim)];
    }
} else {
    gradx  = 0.0;
    grady  = 0.0;
    gradxx = 0.0;
    gradxy = 0.0;
    gradyy = 0.0;
 /*   if(a == xdim-1){
     final[INDEX(a,b,ydim)] = final[INDEX(a-1,b,ydim)];
    }
    if(a== 0){
     final[INDEX(a,b,ydim)] = final[INDEX(a+1,b,ydim)];
    }
    if(b==ydim-1){
     final[INDEX(a,b,ydim)] = final[INDEX(a,b-1,ydim)];     
    }
    if(b==0){
     final[INDEX(a,b,ydim)] = final[INDEX(a,b+1,ydim)];     
    }*/
    
}
    __threadfence();
    image[INDEX(a,b,ydim)] = final[INDEX(a,b,ydim)];
}


} 
"""    
    )


shunshinNoDiffIteration = soussMoKok.get_function("diffIteration")


def shunshinNoStepIteration(dist,timestep,maxit):
    """
    iterates the function image on a 2d grid through an euler anisotropic
    diffusion operator with timestep=timestep maxit number of times
    """
    image = 1*dist
    forme = image.shape
    if(np.size(forme)>2):
        sys.exit('Only works on gray images')

    aSize = forme[0]*forme[1]
    xdim  = np.int32(forme[0])
    ydim  = np.int32(forme[1])  
    
    
    image[0,:]      = image[1,:]
    image[xdim-1,:] = image[xdim-2,:]
    image[:,ydim-1] = image[:,ydim-2]
    image[:,0]      = image[:,1]
     
    image = image.reshape(aSize,order= 'C').astype(np.float32)
    final = np.zeros(aSize).astype(np.float32)
    
    #reshaping the image matrix
    
    
    #block size: B := dim1*dim2*dim3=1024
    #gird size : dim1*dimr2*dim3 = ceiling(aSize/B)
    blockX     = int(1024)
    multiplier = aSize/float(1024)   
    if(aSize/float(1024) > int(aSize/float(1024)) ):
        gridX = int(multiplier + 1)
    else:
        gridX = int(multiplier)
    

    shunshinNoDiffIteration(drv.In(image),drv.Out(final),ydim, xdim, np.float32(timestep), np.float32(maxit),block=(blockX,1,1), grid=(gridX,1,1))


    return final.reshape(forme,order='C')

