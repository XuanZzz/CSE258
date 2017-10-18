# cse 258 hw week2
import numpy
import urllib
import scipy.optimize
import random
from sklearn import svm

def parseData(fname):
  for l in open(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("beer_50000.json"))
print "done"


# 5. SVM -- estimates whether "American IPA"
X = [[d['beer/ABV'], d['review/taste']]for d in data]
y = [d['beer/style']=='American IPA' for d in data]

X_train = X[:len(data)/2]
y_train = y[:len(data)/2]

X_test = X[len(data)/2:]
y_test = y[len(data)/2:]


# Create a support vector classifier object, with regularization parameter C = 1000
clf = svm.SVC(C=1000, kernel = 'linear')
clf.fit(X_train, y_train)

print "Predicting..."
train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)
print "done"

train_match = [(x==y) for x,y in zip(train_predictions, y_train)]
test_match = [(x==y) for x,y in zip(test_predictions, y_test)]
print "Accuracy for training set: ", sum(train_match)*1.0/float(len(train_match))
print "Accuracy for testing set: ", sum(test_match)*1.0/float(len(test_match))

# 6. pick some parameters
# print "Using features of review/appearance and review/taste..."
# X = [[d['review/appearance'], d['review/taste']] for d in data]
# X = [[d['review/appearance'], 'orange' in d['review/text']] for d in data]
X = [[d['review/appearance'], 'IPA' in d['beer/name']] for d in data]
y = [d['beer/style']=='American IPA' for d in data]
f = [False for d in data]
X_train = X[:len(data)/2]
y_train = y[:len(data)/2]

X_test = X[len(data)/2:]
y_test = y[len(data)/2:]

'''
# Create a support vector classifier object, with regularization parameter C = 1000
clf = svm.SVC(C=1000)
clf.fit(X_train, y_train)

print "Predicting..."
train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)
print "done"

train_match = [(x==y) for x,y in zip(train_predictions, y_train)]
test_match = [(x==y) for x,y in zip(test_predictions, y_test)]
print "Accuracy for training set: ", sum(train_match)*1.0/float(len(train_match))
print "Accuracy for testing set: ", sum(test_match)*1.0/float(len(test_match))
'''
# 7. try different regularization constants
constants = [0.1, 10, 1000, 100000]
# constants = [1000]
for c in constants:
	print "C = ", c
	clf = svm.SVC(C=c, kernel = 'linear')
	clf.fit(X_train, y_train)

	train_predictions = clf.predict(X_train)
	test_predictions = clf.predict(X_test)

	train_match = [(x==y) for x,y in zip(train_predictions, y_train)]
	test_match = [(x==y) for x,y in zip(test_predictions, y_test)]
	print "Accuracy for training set: ", sum(train_match)*1.0/float(len(train_match))
	print "Accuracy for testing set: ", sum(test_match)*1.0/float(len(test_match))	

false_match = [(x==y) for x,y in zip(f, y_train)]
print sum(false_match)*1.0/float(len(false_match))
false_match = [(x==y) for x,y in zip(f, y_test)]
print sum(false_match)*1.0/float(len(false_match))
