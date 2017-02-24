# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:34:23 2017

@author: amr62
"""
import os


#lists the files in the directory
filenames    =  os.listdir('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/AR xrays/all Xrays')
patientList  = filenames
N = len(filenames)

for i in range(0,N):
 myfile = filenames[i]

 k          = 0
 word       = myfile[k]
 imagename  = ''
 while(word != '_'):
  imagename = imagename + word   
  k = k+1
  word = myfile[k]
 patientList[i] = imagename
 