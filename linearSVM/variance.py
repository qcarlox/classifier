import numpy
numpy.set_printoptions(threshold=numpy.inf)
filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/data.npz'
npz = numpy.load(filename)
trainingFeatures = npz['features']
featureList = npz['featureList']
print trainingFeatures.shape
STD = numpy.std(trainingFeatures,0)
nonZero = numpy.squeeze(numpy.nonzero(STD))
trainingFeatures = trainingFeatures[:,nonZero]
print trainingFeatures.shape
featureList = featureList[nonZero]
MEAN = numpy.mean(trainingFeatures,0)
STD = numpy.std(trainingFeatures,0)

'''
for k,v in enumerate(MEAN):
    print v

#trainingFeatures = (trainingFeatures-MEAN)/ STD
covMat = numpy.triu(numpy.cov(trainingFeatures.T))
covMat = numpy.triu(numpy.random.rand(4,4))
rows = covMat.shape[0]
cols = covMat.shape[1]
stats = 9
variaceVector = numpy.zeros(stats)
'''


for k,f in enumerate(featureList):
    print (f +",%.15f") % (abs(MEAN[k])/STD[k])

'''
for k in range(rows):
    for l in range(cols):
        for m in range(stats):
            if (l+1)%(stats) == m and (k+1)%(stats) == m:
                if k==l:
                    variaceVector[m] = variaceVector[m] + 2*covMat[k,l]
                else:
                    variaceVector[m] = variaceVector[m]
print variaceVector
'''
