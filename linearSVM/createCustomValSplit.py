def createCustomValSplit():
    import glob
    import numpy
    import os
    splitPaths = [
        '/home/user/Desktop/soundsDB2/datasets/animal/split/trainingData/',
        '/home/user/Desktop/soundsDB2/datasets/bird/split/trainingData/',
        '/home/user/Desktop/soundsDB2/datasets/gun/split/trainingData/',
        '/home/user/Desktop/soundsDB2/datasets/human/split/trainingData/',
        '/home/user/Desktop/soundsDB2/datasets/vehicle/split/trainingData/',
        '/home/user/Desktop/soundsDB2/datasets/noise/split/trainingData/',
        '/home/user/Desktop/soundsDB2/datasets/noise/noisy/trainingData/',
    ]
    filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/sourcePaths.npz'
    npz = numpy.load(filename)
    sourcePaths = npz['sourcePaths']
    
    fileDictionary={}
    key = 0
    for curPath in splitPaths:
        files = sorted(glob.glob(curPath+"*.*"))
        for curFile in files:
            if 'pureNoise' in curFile:
                fileDictionary.update({curFile:key})
            else:
                index = []
                for i,char in enumerate(curFile):
                    if char == '/':
                        index.append(i)
                fileName = curFile[index[-1]+1:]
                fileDictionary.update({fileName:key})
            key = key+1
            
    cvLabels = []
    for curPath in sourcePaths:
        files = sorted(glob.glob(curPath+"*.*"))
        for curFile in files:
            if 'pureNoise' in curFile:
                cvLabels.append(fileDictionary[curFile])
            else:
                index = curFile.find("8000")
                fileName = curFile[index:]
                cvLabels.append(fileDictionary[fileName])

    filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/cvLabels.npz'
    numpy.savez(filename,cvLabels=cvLabels)
    return
