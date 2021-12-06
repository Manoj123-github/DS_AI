

# hides all warnings
import warnings
warnings.filterwarnings('ignore')

# imports
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
# sns
import seaborn as sns
# util
import utils


##############################################################
# Read Data 
##############################################################

# read dataset
print("\n*** Read Data ***")
df = pd.read_csv('./data/glassm-data.csv')
print("Done ...")


##############################################################
# Exploratory Data Analytics
##############################################################

# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())


##############################################################
# Class Variable & Counts
##############################################################

# store class variable  
# change as required
clsVars = "Type"
print("\n*** Class Variable ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# get unique Class names
print("\n*** Unique Class - Categoric Numeric ***")
lnLabels = df[clsVars].unique()
print(lnLabels)

# not required as class var is already numeric


##############################################################
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
df = df.drop('Id', axis=1)
print("Done ...")

# transformations
# change as required
print("\n*** Transformations ***")
df['Ca'] = np.where(df['Ca'].str.strip()=='',None,df['Ca'])
df['Ca'] = df['Ca'].astype(np.float64)
print("Done ...")

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outlier
print('\n*** Handle Outliers ***')
colNames = ['Ka','Ba','Fe']
for colName in colNames:
      colType =  df[colName].dtype  
      df[colName] = utils.HandleOutliers(df[colName])
      if df[colName].isnull().sum() > 0:
          df[colName] = df[colName].astype(np.float64)
      else:
          df[colName] = df[colName].astype(colType)    
print("Done ...")

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if require

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# check mean
print('\n*** Mean In Columns ***')
print(df.mean())

# # normalize data
# print('\n*** Normalize Data ***')
# df = utils.NormalizeData(df, clsVars)
# print('Done ...')

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls
colNames = ['RI','Na','Mg','Al','Si','Ca']
for colName in colNames:
    colType =  df[colName].dtype  
    df[colName] = df[colName].fillna(df[colName].mean())
    df[colName] = df[colName].astype(colType)    
print("Done ...")

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# feature selection
print("\n*** Feature Scores - XTC ***")
print(utils.getFeatureScoresXTC(df, clsVars))

print("\n*** Feature Scores - SKC ***")
print(utils.getFeatureScoresSKB(df, clsVars))

# drop cols
# change as required
print("\n*** Drop Cols ***")
df = df.drop('Ba', axis=1)
print("Done ...")

##############################################################
# Visual Data Anlytics
##############################################################

# boxplot
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
for colName in colNames:
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# histograms
# plot histograms
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
    
# check class
# outcome groupby count    
print("\n*** Group Counts ***")
print(df.groupby(clsVars).size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df[clsVars],label="Count")
plt.title('Class Variable')
plt.show()


################################
# Classification 
# set X & y
###############################

# split into data & target
print("\n*** Prepare Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(clsVars)
print(allCols)
X = df[allCols].values
y = df[clsVars].values

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

# head
print("\n*** Prepare Data - Head ***")
print(X[0:4])
print(y[0:4])

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# over sampling
print("\n*** Over Sampling Process ***")
X, y = utils.getOverSamplerData(X, y)
print("Done ...")

# counts
print("\n*** Counts ***")
unique_elements, counts_elements = np.unique(y, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

################################
# Classification 
# Split Train & Test
###############################

# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                test_size=0.33, random_state=707)

# shapes
print("\n*** Train & Test Data ***")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))

################################
# Classification 
# actual model ... create ... fit ... predict
###############################

# original
# import all model & metrics
print("\n*** Importing Models ***")
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB
print("Done ...")

# create a list of models so that we can use the models in an iterstive manner
print("\n*** Creating Models ***")
lModels = []
lModels.append(('SVM-Clf', SVC(random_state=707)))
lModels.append(('RndFrst', RandomForestClassifier(random_state=707)))
lModels.append(('KNN-Clf', KNeighborsClassifier()))
lModels.append(('LogRegr', LogisticRegression(random_state=707)))
lModels.append(('DecTree', DecisionTreeClassifier(random_state=707)))
lModels.append(('GNBayes', GaussianNB()))
for vModel in lModels:
    print(vModel)
print("Done ...")


################################
# Classification 
# Cross Validation
###############################

# blank list to store results
print("\n*** Cross Validation Init ***")
xvModNames = []
xvAccuracy = []
xvSDScores = []
print("Done ...")

# cross validation
from sklearn import model_selection
print("\n*** Cross Validation ***")
# iterate through the lModels
for vModelName, oModelObj in lModels:
    # select xv folds
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=707)
    # actual corss validation
    cvAccuracy = cross_val_score(oModelObj, X, y, cv=kfold, scoring='accuracy')
    # prints result of cross val ... scores count = lfold splits
    print(vModelName,":  ",cvAccuracy)
    # update lists for future use
    xvModNames.append(vModelName)
    xvAccuracy.append(cvAccuracy.mean())
    xvSDScores.append(cvAccuracy.std())
    
# cross val summary
print("\n*** Cross Validation Summary ***")
# header
msg = "%10s: %10s %8s" % ("Model   ", "xvAccuracy", "xvStdDev")
print(msg)
# for each model
for i in range(0,len(lModels)):
    # print accuracy mean & std
    msg = "%10s: %5.7f %5.7f" % (xvModNames[i], xvAccuracy[i], xvSDScores[i])
    print(msg)

# find model with best xv accuracy & print details
print("\n*** Best XV Accuracy Model ***")
maxXVIndx = xvAccuracy.index(max(xvAccuracy))
print("Index      : ",maxXVIndx)
print("Model Name : ",xvModNames[maxXVIndx])
print("XVAccuracy : ",xvAccuracy[maxXVIndx])
print("XVStdDev   : ",xvSDScores[maxXVIndx])
print("Model      : ",lModels[maxXVIndx])
 

################################
# Classification 
# evaluate : Accuracy & Confusion Metrics
###############################

# print original confusion matrix
print("\n*** Confusion Matrix ***")
cm = confusion_matrix(y_test, y_test)
print("Original")
print(cm)

# blank list to hold info
print("\n*** Confusion Matrix - Init ***")
cmModelInf = []
cmModNames = []
cmAccuracy = []
print("\nDone ... ")

# iterate through the modes and calculate accuracy & confusion matrix for each
print("\n*** Confusion Matrix - Compare ***")
for vModelName, oModelObj in lModels:
    # blank model object
    model = oModelObj
    # fit the model with train dataset
    model.fit(X_train, y_train)
    # predicting the Test set results
    p_test = model.predict(X_test)
    # accuracy
    vAccuracy = accuracy_score(y_test, p_test)
    # confusion matrix
    cm = confusion_matrix(y_test, p_test)
    # X-axis Predicted | Y-axis Actual
    print("")
    print(vModelName)
    print(cm)
    print("Accuracy", vAccuracy*100)
    # update lists for future use 
    cmModelInf.append((vModelName, oModelObj, cmAccuracy))
    cmModNames.append(vModelName)
    cmAccuracy.append(vAccuracy)

# conf matrix summary
print("\n*** Confusion Matrix Summary ***")
# header
msg = "%7s: %10s " % ("Model", "xvAccuracy")
print(msg)
# for each model
for i in range(0,len(cmModNames)):
    # print accuracy mean & std
    msg = "%8s: %5.7f" % (cmModNames[i], cmAccuracy[i]*100)
    print(msg)

print("\n*** Best CM Accuracy Model ***")
maxCMIndx = cmAccuracy.index(max(cmAccuracy))
print("Index      : ",maxCMIndx)
print("Model Name : ",cmModNames[maxCMIndx])
print("CMAccuracy : ",cmAccuracy[maxCMIndx]*100)
print("Model      : ",lModels[maxCMIndx])


################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

print("\n*** Accuracy & Models ***")
print("Cross Validation")
print("Accuracy:", xvAccuracy[maxXVIndx])
print("Model   :", lModels[maxXVIndx]) 
print("Confusion Matrix")
print("Accuracy:", cmAccuracy[maxCMIndx])
print("Model   :", lModels[maxCMIndx]) 

# classifier object
# select best cm acc ... why
print("\n*** Classfier Object ***")
model = lModels[maxCMIndx][1]
print(model)
# fit the model
model.fit(X_train,y_train)
print("Done ...")

# classifier object
print("\n*** Predict Test ***")
# predicting the Test set results
p_test = model.predict(X_test)            # use model ... predict
print("Done ...")

# accuracy
accuracy = accuracy_score(y_test, p_test)*100
print("\n*** Accuracy ***")
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
cm = confusion_matrix(y_test, y_test)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix
# X-axis Predicted | Y-axis Actual
cm = confusion_matrix(y_test, p_test)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_test,p_test)
print(cr)
  
################################
# Final Prediction
# Create Knn Object from whole data
# Read .prd file
# Predict Species
# Confusion matrix with data in .prd file
###############################

# read dataset
print("\n*** Read Data For Prediction ***")
dfp = pd.read_csv('./data/glassm-data-prd.csv')
print(dfp.info())

# not required
print("\n*** Data For Prediction - Drop Cols***")
print("N/A ...")

# convert string / categoric to numeric
print("\n*** Data For Prediction - Class Vars ***")
print(dfp[clsVars].unique())

# change as required ... same transformtion as done for main data
print("\n*** Data For Prediction - Transformation ***")
#dfp = utils.NormalizeData(dfp, clsVars)
print("None ...")

# check nulls
print('\n*** Data For Prediction - Columns With Nulls ***')
print(dfp.isnull().sum()) 
print("Done ... ")

# split into data & outcome
print("\n*** Data For Prediction - X & y Split ***")
print(allCols)
print(clsVars)
X_pred = dfp[allCols].values
y_pred = dfp[clsVars].values
print(X_pred)
print(y_pred)

print("\n*** Accuracy & Models ***")
print("Cross Validation")
print("Accuracy:", xvAccuracy[maxXVIndx])
print("Model   :", lModels[maxXVIndx]) 
print("Confusion Matrix")
print("Accuracy:", cmAccuracy[maxCMIndx])
print("Model   :", lModels[maxCMIndx]) 

# classifier object
# select best cm acc ... why
print("\n*** Classfier Object ***")
model = lModels[maxCMIndx][1]
print(model)
# fit the model
model.fit(X,y)
print("Done ...")

# predict from model
print("\n*** Actual Prediction ***")
p_pred = model.predict(X_pred)
# actual
print("Actual")
print(y_pred)
# predicted
print("Predicted")
print(p_pred)

# accuracy
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_pred, p_pred)*100
print(accuracy)

# confusion matrix - actual
cm = confusion_matrix(y_pred, y_pred)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix - predicted
cm = confusion_matrix(y_pred, p_pred)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_pred, p_pred)
print(cr)

# update data frame
print("\n*** Update Predict Data ***")
dfp['Predict'] = p_pred
print("Done ...")

# save to cls file
print("\n*** Save Data To File **")
df.to_csv('./data/glassm-data-cls.csv', index=False)
print("Done ...")
