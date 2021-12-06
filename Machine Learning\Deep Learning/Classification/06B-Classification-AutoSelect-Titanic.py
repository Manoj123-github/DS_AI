
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
df = pd.read_csv('./data/titanic_train.csv')


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
clsVars = "Survived"
print("\n*** Class Vars ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# get unique Species names
print("\n*** Unique Species - Categoric Numeric***")
lnLabels = df[clsVars].unique()
print(lnLabels)
# not required cause class var is already numeric


##############################################################
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
df = df.drop('PassengerId', axis=1)
df = df.drop('Name', axis=1)
df = df.drop('Ticket', axis=1)
df = df.drop('Cabin', axis=1)
df = df.drop('Embarked', axis=1)
print("Done ...")

# convert Sex - string / categoric to numeric
print("\n*** Unique Sex - Categoric Code ***")
lnLabels = df['Sex'].unique()
print(lnLabels)
from sklearn import preprocessing
leSex = preprocessing.LabelEncoder()
df['Sex'] = leSex.fit_transform(df['Sex'])
lnCCodes = df['Sex'].unique()
print(lnCCodes)

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers if required

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros in fare - using group mean ... Pclass is our group
df['Fare'] = np.where(df['Fare']==0,None, df['Fare'])
df['Fare'] = df.groupby('Pclass')['Fare'].apply(lambda x:x.fillna(x.mean()))
df['Fare'] = df['Fare']

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# check mean
print('\n*** Mean In Columns ***')
print(df.mean())

# handle normalization if required

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required
# handle nulls in age
print("\nColumn: Age")
print("Mean:",df['Age'].mean())
print("Median:",df['Age'].median())
vAge = min(df['Age'].mean(),df['Age'].median())
print("Lower:",vAge)
df['Age'] = df['Age'].fillna(vAge)

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 


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
    
# # dist plots
# # all categoic variables except clsVars
# # change as required
# colNames = ['colName']
# print("\n*** Distribution Plot ***")
# for colName in colNames:
#     plt.figure()
#     sns.countplot(df[colName],label="Count")
#     plt.title(colName)
#     plt.show()

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

# split into X & y
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
# Create model from whole data
# Read .prd file
# Predict clsVars
# Confusion matrix with data in .prd file
###############################

# read dataset
print("\n*** Read Data For Prediction ***")
dfp = pd.read_csv('./data/titanic_test.csv')
print(dfp.info())

# not required
print("\n*** Data For Prediction - Drop Cols***")
print("N/A ...")

# convert string / categoric to numeric
print("\n*** Data For Prediction - Class Vars ***")
dfp['Survived'] = -1
print(dfp[clsVars].unique())

# change as required ... same transformtion as done for main data
print("\n*** Data For Prediction - Transformation ***")
# convert Sex - string / categoric to numeric
print("\n*Sex*")
lnCCodes = dfp['Sex'].unique()
print(lnCCodes)
dfp['Sex'] = leSex.transform(dfp['Sex'])
lnCCodes = df['Sex'].unique()
print(lnCCodes)
# check zeros ... if applicable ... replace with NaN 
print('\n*ColumnsWithZeros*')
print((df==0).sum())
# check nulls
print('\n*ColumnsWithNulls*')
print(dfp.isnull().sum()) 
# handle nulls in age
print('\n*Age*')
dfp['Age'] = np.where(dfp['Age']==0,None, dfp['Age'])
# populate with Age used earlier
dfp['Age'] = dfp['Age'].fillna(vAge)
print(vAge)
# handle nulls in fare
print('\n*Fare*')
dfp['Fare'] = dfp.groupby('Pclass')['Fare'].apply(lambda x:x.fillna(x.mean()))
print('\n*ColumnsWithNulls*')
print(dfp.isnull().sum()) 
print("Done ...")

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

# update data frame
print("\n*** Update Predict Data ***")
dfp['Predict'] = p_pred
print("Done ...")


