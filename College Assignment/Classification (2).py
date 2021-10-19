# -*- coding: utf-8 -*-
"""
Created on Sat May 29 00:22:02 2021

@author: Manoj Yadav
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 5)
import seaborn as sns
import utils

##############################################################
# Read Data 
##############################################################

df = pd.read_csv('./data/breast-cancer-wisconsin.csv')

##############################################################
# Exploratory Data Analytics
##############################################################


print("\n*** Columns ***")
print(df.columns)

print("\n*** Structure ***")
print(df.info())

print("\n*** Summary ***")
print(df.describe())

print("\n*** Head ***")
print(df.head())


##############################################################
# Class Variable & Counts
#############################################################r

clsVars = "diagnosis"
print("\n*** Class Vars ***")
print(clsVars)

print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

print("\n*** Unique Species - Categoric Alpha***")
lnLabels = df[clsVars].unique()
print(lnLabels)

print("\n*** Unique Species - Categoric Alpha to Numeric ***")
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df[clsVars] = le.fit_transform(df[clsVars])
lnCCodes = df[clsVars].unique()
print(lnCCodes)

##############################################################
# Data Transformation
##############################################################

print("\n*** Drop Cols ***")
df = df.drop('id', axis=1)
df = df.drop('Unnamed: 32', axis=1)
print("Done ...")

print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

print('\n*** Columns With Zeros ***')
print((df==0).sum())

print('\n*** Variance In Columns ***')
print(df.var())

print('\n*** StdDev In Columns ***')
print(df.std())

print('\n*** Mean In Columns ***')
print(df.mean())

print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

##############################################################
# Visual Data Anlytics
##############################################################

print('\n*** Boxplot ***')
colNames = df.columns.tolist()
for colName in colNames:
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

print("\n*** Histogram Plot ***")
colNames = df.columns.tolist()
colNames.remove(clsVars)
print('Histograms')
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()
    
print("\n*** Group Counts ***")
print(df.groupby(clsVars).size())
print("")

print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df[clsVars],label="Count")
plt.title('Class Variable')
plt.show()


################################
# Classification 
# set X & y
##############s#################

print("\n*** Prepare Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(clsVars)
print(allCols)
X = df[allCols].values
y = df[clsVars].values

print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

print("\n*** Prepare Data - Head ***")
print(X[0:4])
print(y[0:4])


################################
# Classification 
# Split Train & Test
###############################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                test_size=0.33, random_state=707)

print("\n*** Train & Test Data ***")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
print(np.asarray((unique_elements, counts_elements)))

unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))


################################
# Classification 
# actual model ... create ... fit ... predict
###############################

from sklearn.neighbors import KNeighborsClassifier
print("\n*** Classfier ***")
model = KNeighborsClassifier()
print(model)
model.fit(X_train, y_train)              

################################
# Classification  - Predict Train
# evaluate : Accuracy & Confusion Metrics
###########################k###

print("\n*** Predict Train ***")
p_train = model.predict(X_train)
print("Done ...")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train, p_train)*100
print("\n*** Accuracy ***")
print(accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_train)
print("\n*** Confusion Matrix - Original ***")
print(cm)

cm = confusion_matrix(y_train, p_train)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

from sklearn.metrics import classification_report
print("\n*** Classification Report ***")
cr = classification_report(y_train,p_train)
print(cr)

print("\n*** Recreate Train ***")
dfTrain =  pd.DataFrame(data = X_train)
dfTrain.columns = allCols
dfTrain[clsVars] = y_train
dfTrain['Predict'] = p_train
dfTrain[clsVars] = le.inverse_transform(dfTrain[clsVars])
dfTrain['Predict'] = le.inverse_transform(dfTrain['Predict'])
print("Done ...")


################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

print("\n*** Predict Test ***")
p_test = model.predict(X_test)
print("Done ...")

accuracy = accuracy_score(y_test, p_test)*100
print("\n*** Accuracy ***")
print(accuracy)

cm = confusion_matrix(y_test, y_test)
print("\n*** Confusion Matrix - Original ***")
print(cm)

cm = confusion_matrix(y_test, p_test)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

print("\n*** Classification Report ***")
cr = classification_report(y_test,p_test)
print(cr)

print("\n*** Recreate Test ***")
dfTest =  pd.DataFrame(data = X_test)
dfTest.columns = allCols
dfTest[clsVars] = y_test
dfTest['Predict'] = p_test
dfTest[clsVars] = le.inverse_transform(dfTest[clsVars])
dfTest['Predict'] = le.inverse_transform(dfTest['Predict'])
print("Done ...")

################################
# For Knn Classification Only
# find best k ... how?
# find which K has least error
###############################

print("\n*** Find Best K ***")
lKnnCount = []
lKnnError = []
for i in range(3, 25):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    i_test = knn.predict(X_test)
    vKnnAcc = accuracy_score(y_test, i_test)
    lKnnCount.append(i)
    lKnnError.append(round(1-vKnnAcc,2))
    print('%3d. %7.2f' % (i, 1-vKnnAcc))
print("Done ...")

vBestKnnInd = lKnnError.index(min(lKnnError))
print("\n*** Best K Data ***")
print("Best K Index: ",vBestKnnInd)
print("Best K Value: ",lKnnCount[vBestKnnInd])
print("Best K Error: ",lKnnError[vBestKnnInd])

plt.figure(figsize=(10,5))
sns.pointplot(x=lKnnCount, y=lKnnError, color="b", scale=0.5)
plt.title('K Value v/s Error')
plt.xlabel('K Value')
plt.ylabel('Error')
plt.show()

################################
# Final Prediction
# Create model Object from whole data
# Read .prd file
# Predict Species
# Confusion matrix with data in .prd file
###############################

print("\n*** Classfier Object ***")
model = KNeighborsClassifier()
print(model)
model.fit(X,y)
print("Done ...")

print("\n*** Read Data For Prediction ***")
dfp = pd.read_csv('./data/breast-cancer-wisconsin-prd.csv')
print(dfp.head())

print("\n*** Data For Prediction - Transform***")
dfp = dfp.drop('id', axis=1)
dfp=dfp.drop('Unnamed: 32', axis=1)
print(dfp.info())
print("Done ...")

print("\n*** Data For Prediction - Class Vars ***")
print(dfp[clsVars].unique())
dfp[clsVars] = le.transform(dfp[clsVars])
print(dfp[clsVars].unique())

print("\n*** Data For Prediction - X & y Split ***")
allCols = dfp.columns.tolist()
allCols.remove(clsVars)
X_pred = dfp[allCols].values
y_pred = dfp[clsVars].values
print(X_pred)
print(y_pred)

print("\n*** Prediction ***")
p_pred = model.predict(X_pred)
print("Actual")
print(y_pred)
print("Predicted")
print(p_pred)

print("\n*** Accuracy ***")
accuracy = accuracy_score(y_pred, p_pred)*100
print(accuracy)

cm = confusion_matrix(y_pred, y_pred)
print("\n*** Confusion Matrix - Original ***")
print(cm)

cm = confusion_matrix(y_pred, p_pred)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

print("\n*** Classification Report ***")
cr = classification_report(y_pred, p_pred)
print(cr)

print("\n*** Update Predict Data ***")
dfp['Predict'] = p_pred
dfp[clsVars] = le.inverse_transform(dfp[clsVars])
dfp['Predict'] = le.inverse_transform(dfp['Predict'])
print("Done ...")
