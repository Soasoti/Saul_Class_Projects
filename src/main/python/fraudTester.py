import numpy as np
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split as TTS
from sklearn import linear_model, datasets
from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plot Potentially graph some things?

readFile = open('input/creditFraudData.csv', 'r', encoding ='utf-8')
dataFile = pandas.read_csv(readFile)

print('\n')
print(dataFile.head())
print('\n')

#Begin Decision Tree Implementation
from sklearn.tree import DecisionTreeClassifier

classModel = DecisionTreeClassifier()

#Splits data at random into test and train sets
trainData, testData = TTS(dataFile, test_size = 0.30)
trainMatrix = trainData.as_matrix()

#Populates the training matrix
x = trainMatrix[:,range(0,29)]
y = trainMatrix[:,30]
trainedModel = classModel.fit(x,y)

#Populates the testing matrix
testMatrix = testData.as_matrix()
testX = testMatrix[:,range(0,29)]
testY = testMatrix[:,30]

predictionModel = trainedModel.predict(testX)
predictionModel

#Display an output of the results of the model
print('Classification Report: \n')
print(metrics.classification_report(testY, predictionModel))
print('Confusion Matrix: \n')
print(metrics.confusion_matrix(testY, predictionModel))

print('Max Depth = %s'% str(classModel.max_depth) + '\n')
print('Max Features = %s'% classModel.max_features + '\n')

print('Fraud in Test Data = %d'% len(testData[testData['Class']==1]))
print('Normal in Test Data = %d'% len(testData[testData['Class']==0]))


falsePositiveRate, truePositiveRate, thresholds = roc_curve(testY, predictionModel)
rocAuc = auc(falsePositiveRate, truePositiveRate)
print('AUC = %0.4f'% rocAuc)

