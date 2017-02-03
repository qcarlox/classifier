import pickle
import os
import subprocess
import numpy
from linearSvmTesting import linearSvmTesting
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer

noiseTypes = [0,1,2,3,4,5,6,7]
SNRs = [-20,-15,-10,-5,0,5,10,15,20]

'''
noiseTypes = [0]
SNRs = [-20]
'''

sourcePaths = [
    '/home/user/Desktop/soundsDB2/datasets/animal/noisy/testingData/',
    '/home/user/Desktop/soundsDB2/datasets/bird/noisy/testingData/',
    '/home/user/Desktop/soundsDB2/datasets/gun/noisy/testingData/',
    '/home/user/Desktop/soundsDB2/datasets/vehicle/noisy/testingData/',
    '/home/user/Desktop/soundsDB2/datasets/noise/noisy/testingData/',
]

filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/sourcePaths.npz'
numpy.savez(filename,sourcePaths=sourcePaths)

modelFile = '/home/user/Desktop/soundsDB2/classifier/linearSVM/cleanClassifier.npz'


for noiseType in noiseTypes:
    f = open("results/testResults.txt", "w")
    for SNR in SNRs:
        command = 'octave-cli /home/user/Desktop/soundsDB2/datasetTransfomations/createNoisyTestingSet.m ' + str(SNR) + ' ' + str(noiseType)
        subprocess.check_call(command, shell=True)
        
        predictionLabels, trueLabels = linearSvmTesting(modelFile)
        confMat = confusion_matrix(trueLabels, predictionLabels)
        colSum = numpy.diagflat(1.0/numpy.sum(confMat,1))
        accuracy = numpy.mean(numpy.diag(colSum*(confMat+0.0)))
        print accuracy
        print >> f, SNR
        print >> f, "accuracy, %.4f" % accuracy 
        for j,score in enumerate(confMat):
            for k,num in enumerate(score):
                print >> f,num,
                print >> f,',',
                if (k+1)%len(sourcePaths) == 0:
                    print >> f
    if noiseType == 0:
       os.rename("results/testResults.txt","results/testResultsCleanWhite.txt")
    elif noiseType == 1:
       os.rename("results/testResults.txt","results/testResultsCleanRed.txt")
    elif noiseType == 2:
       os.rename("results/testResults.txt","results/testResultsCleanPink.txt")
    elif noiseType == 3:
       os.rename("results/testResults.txt","results/testResultsCleanBlue.txt")
    elif noiseType == 4:
       os.rename("results/testResults.txt","results/testResultsCleanViolet.txt")
    elif noiseType == 5:
       os.rename("results/testResults.txt","results/testResultsCleanProp.txt")
    elif noiseType == 6:
       os.rename("results/testResults.txt","results/testResultsCleanWind.txt")
    elif noiseType == 7:
       os.rename("results/testResults.txt","results/testResultsCleanPropWind.txt")


sourcePaths = [
    '/home/user/Desktop/soundsDB2/datasets/animal/noisy/testingData/',
    '/home/user/Desktop/soundsDB2/datasets/bird/noisy/testingData/',
    '/home/user/Desktop/soundsDB2/datasets/gun/noisy/testingData/',
    '/home/user/Desktop/soundsDB2/datasets/vehicle/noisy/testingData/',
    '/home/user/Desktop/soundsDB2/datasets/noise/noisy/testingData/',
]

filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/sourcePaths.npz'
numpy.savez(filename,sourcePaths=sourcePaths)

modelFile = '/home/user/Desktop/soundsDB2/classifier/linearSVM/noisyClassifier.npz'


for noiseType in noiseTypes:
    f = open("results/testResults.txt", "w")
    for SNR in SNRs:
        command = 'octave-cli /home/user/Desktop/soundsDB2/datasetTransfomations/createNoisyTestingSet.m ' + str(SNR) + ' ' + str(noiseType)
        subprocess.check_call(command, shell=True)
        
        predictionLabels, trueLabels = linearSvmTesting(modelFile)
        confMat = confusion_matrix(trueLabels, predictionLabels)
        colSum = numpy.diagflat(1.0/numpy.sum(confMat,1))
        accuracy = numpy.mean(numpy.diag(colSum*(confMat+0.0)))
        print accuracy
        print >> f, SNR
        print >> f, "accuracy, %.4f" % accuracy 
        for j,score in enumerate(confMat):
            for k,num in enumerate(score):
                print >> f,num,
                print >> f,',',
                if (k+1)%len(sourcePaths) == 0:
                    print >> f
    if noiseType == 0:
       os.rename("results/testResults.txt","results/testResultsNoisyWhite.txt")
    elif noiseType == 1:
       os.rename("results/testResults.txt","results/testResultsNoisyRed.txt")
    elif noiseType == 2:
       os.rename("results/testResults.txt","results/testResultsNoisyPink.txt")
    elif noiseType == 3:
       os.rename("results/testResults.txt","results/testResultsNoisyBlue.txt")
    elif noiseType == 4:
       os.rename("results/testResults.txt","results/testResultsNoisyViolet.txt")
    elif noiseType == 5:
       os.rename("results/testResults.txt","results/testResultsNoisyProp.txt")
    elif noiseType == 6:
       os.rename("results/testResults.txt","results/testResultsNoisyWind.txt")
    elif noiseType == 7:
       os.rename("results/testResults.txt","results/testResultsNoisyPropWind.txt")



