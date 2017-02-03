from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LabelKFold
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import numpy
from timeit import default_timer as timer
#f = open('store.pckl', 'w')
#pickle.dump(object, f)
#f.close()
#filehandler = open('clf.pickle', 'w') 
start = timer()
filename ='/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/frameSize.npz'
npz = numpy.load(filename)
frameSize = npz['frameSize']

filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/data.npz'
npz = numpy.load(filename)
trainingLabels = npz['labels']
trainingFeatures = npz['features']

filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/cvLabels.npz'
npz = numpy.load(filename)
cvLabels = npz['cvLabels']

filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/pcaComponents.npz'
npz = numpy.load(filename)
n_components = [npz['pcaComponents']]
X_train = trainingFeatures
y_train = trainingLabels

print trainingFeatures.shape


folds = 10
scaler = preprocessing.StandardScaler()
pca = PCA()
'''
classifier = RandomForestClassifier()
pipe = Pipeline([('scale', scaler),('classifier', classifier)])
n_estimators = [500]
tuned_parameters = [{'classifier__n_estimators': n_estimators}]
'''

classifier = RandomForestClassifier()
pipe = Pipeline([('scale', scaler), ('pca',pca),('classifier', classifier)])
n_estimators = [500]

tuned_parameters = [{'classifier__n_estimators': n_estimators,'pca__n_components': n_components}]


permute = numpy.random.permutation(len(cvLabels))
cvLabels = cvLabels[permute]
trainingLabels = trainingLabels[permute]
trainingFeatures = trainingFeatures[permute,:]


print("# Tuning hyper-parameters for accuracy")
print()
clf = GridSearchCV(pipe, tuned_parameters, cv = LabelKFold(cvLabels,10), scoring='accuracy',n_jobs=2)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std(), params))
end = timer()


print(end - start)


filename = '/home/user/Desktop/soundsDB2/classifier/linearSVM/model.npz'
numpy.savez(filename,clf=clf,frameSize=frameSize)

