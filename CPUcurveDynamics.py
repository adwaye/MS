import numpy as np


def Sfunc(x,eps):
   #smoothing function that 
   return x/np.sqrt(x**2 + eps**2)

#we want to evolve phi_t = S(Phi_0)(1-|grad phi|)
def distance_dynamics(initial,eps,maxit,tol):
   forme = initial.shape
   Idim  = forme[0]
   Jdim  = forme[1]
   
   newState = 1*initial
   for k in range(0,maxit):
      newstate[1:Idim-2,1:Jdim-2] = newstate[1:Idim-2,1:Jdim-2] + Sfunc(initial)*(1 \
                                    - np.sqrt( (0.5*(newstate[2:Idim-1,1:Jdim-2]-newstate[0:Idim-3,1:Jdim-2]) )**2  \
                                    + (0.5*(newstate[2:Idim-1,2:Jdim-1]-newstate[1:Idim-2,0:Jdim-3]) )**2 )
                                    \)
     
      newstate[0,:]      = newstate[1,:]
      newstate[:,0]      = newstate[:,1]
      newstate[Idim-1,:] = newstate[Idim-2,:]
      newstate[:,Jdim-1] = newstate[:,Jdim-2] 
