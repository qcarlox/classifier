from featureExtractFile import featureExtractFile
import numpy
import sys
ID = sys.argv[1]
filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/temp/chunk'+str(ID)+'.npz'
npz = numpy.load(filename)
chunk = npz['chunk']
print 'chunk'
for curFile in chunk:
    featureExtractFile(curFile)
