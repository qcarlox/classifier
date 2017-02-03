import pickle
import os
import subprocess
import numpy
from createCustomValSplit import createCustomValSplit
from timeit import default_timer as timer
start = timer()

command = 'octave-cli /home/user/Desktop/soundsDB2/datasetTransfomations/createNoisyTrainingSet.m '
subprocess.check_call(command, shell=True)

sourcePaths = [
    '/home/user/Desktop/soundsDB2/datasets/animal/split/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/bird/split/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/gun/split/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/vehicle/split/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/noise/trainingData/',
]


filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/sourcePaths.npz'
numpy.savez(filename,sourcePaths=sourcePaths)

frameSizes = [100,200,400,800,1600]
command = 'echo start > results/results.txt'
subprocess.check_call(command, shell=True)
for frameSize in frameSizes:
    filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/frameSize.npz'
    numpy.savez(filename,frameSize=frameSize)
    command = 'python /home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/featureExtract.py > log.txt'
    subprocess.check_call(command, shell=True)
    command = "echo "+"\""+str(frameSize)+"\n\" >> results/results.txt"
    subprocess.check_call(command, shell=True)
    createCustomValSplit()
    command = 'python linearSvmCv.py >> results/results.txt'
    subprocess.check_call(command, shell=True)
    
    
os.rename("results/results.txt","results/resultsCVClean.txt")
os.rename("model.npz","cleanClassifier.npz")



sourcePaths = [
    '/home/user/Desktop/soundsDB2/datasets/animal/noisy/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/bird/noisy/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/gun/noisy/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/vehicle/noisy/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/noise/trainingData/',
]

filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/sourcePaths.npz'
numpy.savez(filename,sourcePaths=sourcePaths)

frameSizes = [100,200,400,800,1600]
command = 'echo start > results/results.txt'
subprocess.check_call(command, shell=True)
for frameSize in frameSizes:
    filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/frameSize.npz'
    numpy.savez(filename,frameSize=frameSize)
    command = 'python /home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/featureExtract.py > log.txt'
    subprocess.check_call(command, shell=True)
    command = "echo "+"\""+str(frameSize)+"\n\" >> results/results.txt"
    subprocess.check_call(command, shell=True)
    createCustomValSplit()
    command = 'python linearSvmCv.py >> results/results.txt'
    subprocess.check_call(command, shell=True)
    
os.rename("results/results.txt","results/resultsCVNoisy.txt")
os.rename("model.npz","noisyClassifier.npz")


end = timer()
print(end - start)

