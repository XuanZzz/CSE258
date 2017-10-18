import numpy
import urllib
import scipy.optimize
import random
from math import exp
from math import log

def parseData(fname):
  for l in open(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("beer_50000.json"))
print "done"

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  print "ll =", loglikelihood
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0.0]*len(theta)
  for i in range(len(X)):
    sigit = inner(X[i], theta)
    # Fill in code for the derivative
    for k in range(len(dl)):
      dl[k]+=(1-sigmoid(sigit))*X[i][k]
      if not y[i]:
        dl[k] -= X[i][k]
  for k in range(len(dl)):
    dl[k] -= 2 * lam * theta[k]
  # Negate the return value since we're doing gradient *ascent*
  return numpy.array([-x for x in dl])


# Extract features and labels from the data
#X = [[d['beer/ABV'], d['review/taste']] for d in data]
X = [[1, d['beer/ABV'], d['review/taste']] for d in data]
y = [d['beer/style']=='American IPA' for d in data]

X_train = X[:len( X)/2]
y_train = y[:len(y)/2]
X_test = X[len(X)/2:]
y_test = y[len(y)/2:]
# If we wanted to split with a validation set:
#X_valid = X[len(X)/2:3*len(X)/4]
#X_test = X[3*len(X)/4:]

# Use a library function to run gradient descent (or you can implement yourself!)
theta,l,info = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, args = (X_train, y_train, 1.0))
print "Final log likelihood =", -l
print "Theta: ", theta

y_train_pred = [(inner(x,theta)>0) for x in X_train]
train_match = [(x==y) for x,y in zip(y_train, y_train_pred)]
y_test_pred = [inner(x,theta)>0 for x in X_test]
test_match = [(x==y) for x,y in zip(y_test, y_test_pred)]
# Compute the accuracy
print "Accuracy of training set = ", sum(train_match)*1.0 / len(train_match) 
print "Accuracy of testing set = ", sum(test_match)*1.0 / len(test_match) 
print sum(y_train_pred)
print sum(y_test_pred)