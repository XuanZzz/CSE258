# cse 258 hw week1
import numpy
def parseData(fname):
  for l in open(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("beer_50000.json"))
print "done"

#1. lines of reviews and average value for each style
print "#1 num_reviews and average value for each style: "
data_style = {}
for d in data:
  if(d['beer/style'] in data_style):
    data_style[d['beer/style']][0] += 1
    data_style[d['beer/style']][1] += d['review/taste']
  else:
    data_style[d['beer/style']]=[1, d['review/taste']]

print "Style\tTotal reviews\tAverage value"

for style in data_style:
  print style, "\t", data_style[style][0], "\t", data_style[style][1]/float(data_style[style][0])


# 2. Predictor between 'review/taste' and whether it's 'American IPA'
print
print "#2 Predict between 'review/taste' and 'American IPA': "
def feature(datum):
  feat = [1]
  if(datum['beer/style']=='American IPA'):
   feat.append(1)
  else:
   feat.append(0)
  return feat

X = [feature(d) for d in data]
y = [d['review/taste'] for d in data]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
print "Theta: ", theta, ".\n"

#3. train on first half, test on first and second half
data1 = data[:len(data)/2]
data2 = data[len(data)/2:]
print "#3 train on first half, test on second half: "
X1=[feature(d) for d in data1]
y1=[d['review/taste'] for d in data1]

theta,residuals,rank,s = numpy.linalg.lstsq(X1, y1)
print "Theta: ", theta, "."
print "MSE for training data: ", residuals/float(len(data1)), "."

# MSE on second half of data
X2 = [feature(d) for d in data2]
y2 = [d['review/taste'] for d in data2]
X2 = numpy.matrix(X2).T
y_pred2 = theta*X2
mse2 = numpy.square(y_pred2-y2).mean()
print "MSE for test data: ", mse2, ".\n"

#4. train for every style with >= 50 reviews
print "#4, train for every style with >=50 reviews: "
styles = []
for s in data_style:
  if(data_style[s][0]>=50):
    styles.append(s)
print "Styles of over 50 reviews: "
print styles
print 

def feature(datum):
  feat = [1]
  for s in styles:        
    if(datum['beer/style'] == s):
      feat.append(1)
    else:
      feat.append(0) 
  return feat

X_style1=[feature(d) for d in data1]
y_style1=[d['review/taste'] for d in data1]

theta,residuals,rank,s = numpy.linalg.lstsq(X_style1, y_style1)
print "Theta: ", theta, "."
print "MSE for training data: ", residuals/float(len(data1)), "."

# MSE on second half of data
X_style2 = [feature(d) for d in data2]
y_style2 = [d['review/taste'] for d in data2]
X_style2 = numpy.matrix(X_style2).T
y_pred_style2 = theta*X_style2
mse_style2 = numpy.square(y_pred_style2-y_style2).mean()
print "MSE for test data: ", mse_style2, "."
