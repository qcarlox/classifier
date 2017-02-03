import numpy
stats = ['min', 'max', 'median', 'mean', 'var','skew','kurt','dmean', 'dvar']
features = [
        'barkbands',
        'mfcc'
        ]
numOfBarkbands = 27
numOfMfcc = 13



k = 0
featuresToRemove = []
for feature in features:
    if 'barkbands' == feature:
        for j in range(numOfBarkbands):
            for l in range(len(stats)):
                if (j+1 == 16 or j+1 == 19 or j+1 == 22 or j+1 == 4 or j+1 == 10 or j+1 == 7 
                or j+1 == 25 or j+1 == 13 or j+1 == 1):
                    featuresToRemove.append(k)
                k=k+1
    elif 'mfcc' == feature:
        for j in range(numOfMfcc):
            for l in range(len(stats)):
                if (j+1 == -1):
                    featuresToRemove.append(k)
                k=k+1
           
filename = '/home/user/Desktop/soundsDB2/classifier/linearSVM/featuresToRemove.npz'
numpy.savez(filename,featuresToRemove=featuresToRemove)
