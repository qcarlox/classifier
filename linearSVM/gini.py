import numpy
stats = ['min', 'max', 'median', 'mean', 'var', 'dmean', 'dvar']
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
numOfBarkbands = 27
numOfMfcc = 13

filename = '/home/user/Desktop/soundsDB2/classifier/linearSVM/giniList.npz'
npz = numpy.load(filename)
giniList = npz['giniList']

featureGini = 0
featureGiniList = []
for k,giniCoeff in enumerate(giniList):
    print giniCoeff
    featureGini = featureGini + giniCoeff
    if (k+1) % len(stats) == 0:
        featureGiniList.append(featureGini)
        featureGini = 0

k = 0
for feature in features:
    if 'barkbands' == feature:
        for j in range(numOfBarkbands):
            print ("barkbands %d, %.5f" %( j+1, featureGiniList[k]))
            k=k+1
    elif 'mfcc' == feature:
        for j in range(numOfMfcc):
           print ("mfcc %d, %.5f" %( j+1, featureGiniList[k]))
           k=k+1
    else:
        print (feature + ", %.5f" %(featureGiniList[k]))
        k=k+1

k = 0
featuresToRemove = []
for feature in features:
    if 'barkbands' == feature:
        for j in range(numOfBarkbands):
            for l in range(len(stats)):
                if not(j+1 == 22 or j+1 == 3 or j+1 == 7 or j+1 == 10 or j+1 == 15 or j+1 == 6 
                or j+1 == 14 or j+1 == 27 or j+1 == 25 or j+1 == 13 or j+1 == 26):
                    featuresToRemove.append(k)
                k=k+1
    elif 'mfcc' == feature:
        for j in range(numOfMfcc):
            for l in range(len(stats)):
                if not(j+1 == 7 or j+1 == 4 or j+1 == 2 or j+1 == 1 or j+1 == 5 or j+1 == 6):
                    featuresToRemove.append(k)
                k=k+1
           
filename = '/home/user/Desktop/soundsDB2/classifier/linearSVM/featuresToRemove.npz'
numpy.savez(filename,featuresToRemove=featuresToRemove)

