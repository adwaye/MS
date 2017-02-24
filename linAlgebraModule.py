# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:40:51 2017

@author: amr62
"""
#===================================================
#=== Functions in this module:
#=== matAdd(A,B,alpha,beta):  A = alpha*A + beta*B
#===================================================

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import sys
import time

#===================================================
#=== matrix addition                  ==============
#===================================================
#===  A = alpha*A + beta*B             
#=== input A, B, alpha, beta
#=== output A
#===================================================

mod = SourceModule \
    (
        """
#include<stdio.h>
#include<math.h>
#define INDEX(a, b, yshape) (a)*(yshape) + (b)

__global__ void matrixAddition(float *A,float *B, float alpha, float beta, int ydim)
{
unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
  
  unsigned int a      = idx/ydim;
  unsigned int b      = idx%ydim;
  
  A[INDEX(a, b, ydim)] = alpha*A[INDEX(a, b, ydim)]+beta*B[INDEX(a, b, ydim)];

}
"""
    )


matrixAddition = mod.get_function("matrixAddition")


def matAdd(A,B,alpha,beta):
    forme1 = A.shape
    forme2 = B.shape
    if(forme1 != forme2):
        sys.exit('matrix dimensions differ')
        
        
    aSize  = forme1[0]*forme1[1]
    xdim   = np.int32(forme1[0])
    ydim   = np.int32(forme1[1])
    
    A     = np.reshape(A,aSize,order='F').astype(np.float32)
    B     = np.reshape(B,aSize,order='F').astype(np.float32)
    alpha = np.float32(alpha)
    beta  = np.float32(beta)
    
    blockX = int(ydim)
    gridX  = int(xdim)
    

    matrixAddition(drv.InOut(A),drv.In(B),alpha,beta, ydim, block= (blockX,1,1), grid= (gridX,1,1))
    A = np.reshape(A,forme1,order='F')
    
    return A
    


