# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:34:23 2017

@author: amr62
"""
import os
import pandas

#############################################################################
# gives 2 lists, one with filenames of images called filenames
# other with patient ids called patientList
#############################################################################

#lists the files in the directory
filenames    =  os.listdir('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/AR xrays/all Xrays')

N = len(filenames)

patientList  = [None]*N

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


nIterationList = [None]*N


for n in range(0,len(patientList)):
    patientID = patientList[n]
    
    order66  = "cd /home/amr62/fastms; ./main -edges 0 -i /home/amr62/Documents/TheEffingPhDHatersGonnaHate/AR\ xrays/all\ Xrays/"+patientID+"_1_0.jpg -alpha -1 -lambda 0.25 -show 0 -save '/' >> /home/amr62/Documents/TheEffingPhDHatersGonnaHate/fastmsToying/"+patientID+"result.txt"
    os.system(order66) 
    print('I\'m done bicthes (fastms has terminated) for '+patientID )
    #counting number of  new lines
    filestr    = '/home/amr62/Documents/TheEffingPhDHatersGonnaHate/fastmsToying/'+patientID+'result.txt'
    resultfile = open(filestr,'r')  

    resultString = resultfile.read()


    minbar = int(0 )
    maxbar = int(10)
    newLine = 0
    for j in range(0,len(resultString)):
        if resultString[j] == '\n':
            newLine = newLine +1
            if newLine == 25:
                rasengan = 0
                tempWord = resultString[j+minbar:j+maxbar]   
                while rasengan==0:
                    minbar = minbar+1
                    maxbar = maxbar+1
                    tempWord = resultString[(j+minbar):(j+maxbar)]   
                    #print(tempWord)
                    if(tempWord =='iterations'):
                        rasengan = 1
                        wordLoc  = j+minbar

    backtrack = 2
    while resultString[wordLoc-backtrack] != ' ':
        backtrack = backtrack+1 
    print(resultString[(wordLoc-backtrack+1):(wordLoc-1)]+tempWord)

    nIterationList[n] = int(resultString[(wordLoc-backtrack+1):(wordLoc-1)])




#############################################################################
# data file cleaning: score tabulation
#############################################################################

ratingen = pandas.read_csv('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/AR xrays/More severe cases/Copy of WT-xray-Ratingen (2).csv')


M        = ratingen.shape[0]
year     = [None]*M


for k in range(0,M):
    year[k] = '20'+ratingen['Date'][k][6:8]


colnames = ratingen.columns

def extractPatientName(patientID):
    k          = 0
    word       = patientID[k]
    rasengan   = 0
    while(rasengan == 0):
        k = k+1        
        if(word == 'h'):
            rasengan = 1
        if(word == 'f'):
            rasengan = 1
        word = myfile[k]
        patientList[i] = imagename
        
    patientNumber = patientID[0:(k-1)]
    return patientNumber


iterationVscore = pandas.read_csv('/home/amr62/Documents/TheEffingPhDHatersGonnaHate/fastmsToying/trialimages/ratingen_hand.csv')





plt.subplots_adjust(bottom = 0.1)
plt.scatter(
    iterationVscore['Iterations'],iterationVscore['Score'], marker='o',
    cmap=plt.get_cmap('Spectral'))

for label, x, y in zip(iterationVscore['PatientID'],iterationVscore['Iterations'],iterationVscore['Score']):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.xlabel('Itearation Number')
plt.ylabel('Ratingen Hand score')
plt.show()

plt.plot(iterationVscore['Iterations'],iterationVscore['Score'],"o")
plt.show()
















