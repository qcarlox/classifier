def linearSvmTesting(modelFile):
    
    from sklearn.decomposition import PCA
    from sklearn.svm import LinearSVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.cross_validation import LabelKFold
    from sklearn.feature_selection import RFECV
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    import scipy.io as sio
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    from sklearn.preprocessing import Imputer
    from sklearn import preprocessing
    import numpy
    import subprocess

    npz = numpy.load(modelFile)
    clf = npz['clf'].item()
    frameSize = npz['frameSize']

    filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/frameSize.npz'
    numpy.savez(filename,frameSize=frameSize)

    command = 'python /home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/featureExtract.py > log.txt'
    
    subprocess.check_call(command, shell=True)
    filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/data.npz'
    npz = numpy.load(filename)
    testingFeatures = npz['features']
    trueLabels = npz['labels']
    X_train = testingFeatures
    predictionLabels = clf.predict(X_train)
    
    return predictionLabels, trueLabels

