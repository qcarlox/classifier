import pickle
import os
import subprocess
import numpy
from createCustomValSplit import createCustomValSplit
from timeit import default_timer as timer
start = timer()
'''
sourcePaths = [
    '/home/user/Desktop/soundsDB2/datasets/animal/noisy/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/bird/noisy/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/gun/noisy/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/vehicle/noisy/trainingData/',
]
'''

sourcePaths = [
    '/home/user/Desktop/soundsDB2/datasets/signal/trainingData/',
    '/home/user/Desktop/soundsDB2/datasets/noise/trainingData/',
]

filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/sourcePaths.npz'
numpy.savez(filename,sourcePaths=sourcePaths)

frameSizes = [400]
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
    
end = timer()
print(end - start)   
