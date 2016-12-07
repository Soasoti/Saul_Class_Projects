import numpy as np
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

readFile = open('input/creditFraudData.csv', 'r', encoding ='utf-8')
dataFile = pandas.read_csv(readFile)

print('\n')
print('Head of Datafile:')
print(dataFile.head())
print('\n')

featureSize = len(dataFile.columns)
classIndex  = featureSize - 1

fraudBase  = dataFile[dataFile['Class'] == 1]
normalBase = dataFile[dataFile['Class'] == 0]

#Undersamples the normal transactions due to sheer numbers compared to fraud transactions

percentFraud = len(fraudBase)/float(len(normalBase))

normalTrans  = normalBase.sample(frac = percentFraud)
fraudTrans   = fraudBase

underData    = fraudTrans.append(normalTrans) 

#Splits data at random into test and train sets
trainData, testData = train_test_split(dataFile, test_size = 0.30)
trainMatrix = trainData.as_matrix()
testMatrix  = testData.as_matrix()
#Populates the training matrix
x = trainMatrix[:,range(0,classIndex-1)]
y = trainMatrix[:,classIndex]

#Populates the testing matrix
testX = testMatrix[:,range(0,classIndex-1)]
testY = testMatrix[:,classIndex]

####### Begin Decision Tree Implementation #######
from sklearn.tree import DecisionTreeClassifier

print('Basic Decision Tree:')
print('\n')

treeModel = DecisionTreeClassifier()

trainedTreeModel = treeModel.fit(x,y)

treePredModel = trainedTreeModel.predict(testX)
treePredModel

#Display an output of the results of the model
print(metrics.classification_report(testY, treePredModel))
print(metrics.confusion_matrix(testY, treePredModel))
print('\n')

print('Max Depth = %s'% str(treeModel.max_depth) + '\n')
print('Max Features = %s'% treeModel.max_features + '\n')
print('\n')

print('Fraud in Test Data = %d'% len(testData[testData['Class']==1]))
print('Normal in Test Data = %d'% len(testData[testData['Class']==0]))
print('\n')

treeFPR, treeTPR, treeThresholds = roc_curve(testY, treePredModel)
treeRocAuc = auc(treeFPR, treeTPR)
print('AUC = %0.4f'% treeRocAuc)
print('\n')

plt.title('Receiver Operating Characteristic')
plt.plot(treeFPR, treeTPR, 'b',
label='AUC = %0.2f'% treeRocAuc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

####### Begin Random Forest Implementation #######

from sklearn.ensemble import RandomForestClassifier

print('Random Forest:')
print('\n')

forestModel = RandomForestClassifier(criterion = 'entropy', n_estimators = 100)

forestModel = forestModel.fit(x,y)

forestPredModel = forestModel.predict(testX)
forestPredModel

#Display an output of the results of the model
print(metrics.classification_report(testY, forestPredModel))
print(metrics.confusion_matrix(testY, forestPredModel))
print('\n')

print('Max Depth = %s'% str(forestModel.max_depth))
print('Max Features = %s'% forestModel.max_features)
print('\n')

print('Fraud in Test Data = %d'% len(testData[testData['Class']==1]))
print('Normal in Test Data = %d'% len(testData[testData['Class']==0]))
print('\n')

forestFPR, forestTPR, forestThresholds = roc_curve(testY, forestPredModel)
forestRocAuc = auc(forestFPR, forestTPR)
print('AUC = %0.4f'% forestRocAuc)
print('\n')

plt.title('Receiver Operating Characteristic')
plt.plot(forestFPR, forestTPR, 'b',
label='AUC = %0.2f'% forestRocAuc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
