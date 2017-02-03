from featureExtractDirs import featureExtractDirs
import numpy
import os
import subprocess
filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/sourcePaths.npz'
npz = numpy.load(filename)
sourcePaths = npz['sourcePaths']
for path in sourcePaths:
    command = 'rm ' + path + '*.yaml'
    subprocess.call(command, shell=True)

labels, features, featureList = featureExtractDirs(sourcePaths)
filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/data.npz'
numpy.savez(filename,labels=labels,features=features,featureList=featureList)

