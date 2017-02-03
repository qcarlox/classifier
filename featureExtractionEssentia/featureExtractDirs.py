def featureExtractDirs(paths):
    features = [
        'barkbands',
        'mfcc',
        'barkbands_kurtosis',
        'barkbands_skewness',
        'barkbands_spread',
        'hfc',
        'pitch',
        'pitch_instantaneous_confidence',
        'pitch_salience',
        'silence_rate_20dB',
        'silence_rate_30dB',
        'silence_rate_60dB',
        'spectral_complexity',
        'spectral_crest',
        'spectral_decrease',
        'spectral_energy',
        'spectral_energyband_low',
        'spectral_energyband_middle_low',
        'spectral_energyband_middle_high',
        'spectral_flatness_db',
        'spectral_flux',
        'spectral_rms',
        'spectral_rolloff',
        'spectral_strongpeak',
        'zerocrossingrate',
        ]
        
        
    #stats = ['min','var','skew','kurt', 'dvar']
    stats = ['min', 'max', 'median', 'mean', 'var','skew','kurt','dmean', 'dvar']
    import glob
    import sys
    import yaml
    import pprint
    import numpy
    import os
    import subprocess
    import pickle
    from multiprocessing import Pool
    #f = open('trainingData.pckl', 'w')
    #pickle.dump([trainingLabels, trainingFeatures], f)
    #f.close()
    
    chunkSize = 100
    for curPath in paths:
        files = sorted(glob.glob(curPath+"*.*"))
        ID = 0
        concurrent = 0
        procs = []
        for i in range(0, len(files), chunkSize):
            chunk = files[i:i + chunkSize]
            filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/temp/chunk'+str(ID)+'.npz'
            numpy.savez(filename,chunk=chunk)
            p = subprocess.Popen(['python', '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/featureExtractChunk.py',str(ID)])
            ID = ID + 1
            procs.append(p)
            concurrent = concurrent +1
            if concurrent == 4:
                for p in procs:
                    p.wait()
                concurrent = 0
                procs = []
        for p in procs:
            p.wait()
    featureList = []
    featureMatrix = []
    labelVector = []
    i=0
    flag=1
    for curPath in paths:
        files = glob.glob(curPath+"*.yaml")
        for curFile in files:
            f = open(curFile)
            yamlDic = yaml.safe_load(f)
            data = yamlDic[curFile[:-21]]
            featureVector = numpy.array([] , dtype = float)
            for feature in features:
                dataFeature = data[feature]
                for stat in stats:
                    temp = numpy.asarray(dataFeature[stat], dtype = float)
                    featureVector = numpy.append(featureVector,temp)
                    if flag == 1:
                        if temp.size > 1:
                            for k in range(len(temp)):
                                featureList.append(feature+" "+str(k+1)+","+stat)
                        else:
                            featureList.append(feature+","+stat)
            flag = 0
            featureVector[~numpy.isfinite(featureVector)] = 0
            labelVector.append(i)
            featureMatrix.append(featureVector)
            f.close()
            os.remove(curFile) 
        i = i+1
    return labelVector, featureMatrix, featureList
