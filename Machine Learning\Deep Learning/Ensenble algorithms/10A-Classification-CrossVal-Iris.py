

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
df = pd.read_csv('C:/Users/Manoj Yadav/Desktop/Part II/ML II/Ensenble algorithms/Iris.csv')
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
clsVars = "Species"
print("\n*** Class Vars ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# get unique Species names
print("\n*** Unique Species - Categoric Alpha***")
lnLabels = df[clsVars].unique()
print(lnLabels)

# convert string / categoric to numeric
print("\n*** Unique Species - Categoric Alpha to Numeric ***")
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df[clsVars] = le.fit_transform(df[clsVars])
lnCCodes = df[clsVars].unique()
print(lnCCodes)


##############################################################
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
df = df.drop('Id', axis=1)
print("Done ...")

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

# handle zeros if required

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


##############################################################
# Visual Data Anlytics
##############################################################

# pair plot
print("\n*** Pair Plot ***")
plt.figure()
sns.pairplot(df, height=2)
plt.show()

# boxplot
print("\n*** Box Plot ***")
plt.figure(figsize=(10,5))
sns.boxplot(data=df)
plt.show()

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
    
# dist plots
# all categoic variables except clsVars
# change as required
# colNames = ['Pclass','Sex','SibSp','Parch']
# print("\n*** Distribution Plot ***")
# for colName in colNames:
#     plt.figure()
#     sns.countplot(df[colName],label="Count")
#     plt.title(colName)
#    plt.show()
 
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
# Classification - set X & y
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
# Classification - init models
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

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier


#Gradient Boosting Classifier
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
from sklearn.ensemble import GradientBoostingClassifier

#Ada Boost Classifier
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier


#XGBoost Classifier
# https://xgboost.readthedocs.io/en/latest/
import xgboost as xgb
# booster = gblinear (regression), gbteee (classification) 
# objective = reg:squarederror (regression), binary:logistic (binary), multi:softprob (multiclass)

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
lModels.append(('RandomForestClassifier',RandomForestClassifier(random_state=707)))
lModels.append(('GradientBoostingClassifier',GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=707)))
lModels.append(('AdaBoostClassifier',AdaBoostClassifier(n_estimators=100, random_state=707)))
lModels.append(('XGBoostClassifier',xgb.XGBClassifier(booster='gbtree', objective='multi:softprob', verbosity=0, seed=707)))
for vModel in lModels:
    print(vModel)
print("Done ...")


################################
# Classification - cross validation
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
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=707)
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
xvIndex = xvAccuracy.index(max(xvAccuracy))
print("Index      : ",xvIndex)
print("Model Name : ",xvModNames[xvIndex])
print("XVAccuracy : ",xvAccuracy[xvIndex])
print("XVStdDev   : ",xvSDScores[xvIndex])
print("Model      : ",lModels[xvIndex])

###############################
# Split Train & Test
# Classification 
###############################

# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                test_size=0.2, random_state=707)

# shapes
print("\n*** Train & Test Data ***")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
################################
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))


################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

print("\n*** Accuracy & Models ***")
print("Cross Validation")
print("Accuracy:", xvAccuracy[xvIndex])
print("Model   :", lModels[xvIndex]) 

# classifier object
# select model with best acc
print("\n*** Classfier Object ***")
model = lModels[xvIndex][1]
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
# Predict cldVars
# Confusion matrix with data in .prd file
#######N#######################

# read dataset
print("\n*** Read Data For Prediction ***")
dfp = pd.read_csv('C:/Users/Manoj Yadav/Desktop/Part II/ML II/Ensenble algorithms/iris-prd.csv')
print(dfp.head())

# not required
print("\n*** Data For Prediction - Drop Cols***")
print("N/A ...")

# convert string / categoric to numeric
print("\n*** Data For Prediction - Class Vars ***")
print(dfp[clsVars].unique())
dfp[clsVars] = le.transform(dfp[clsVars])
print(dfp[clsVars].unique())

# change as required ... same transformtion as done for main data
print("\n*** Data For Prediction - Transformation ***")
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
print("Accuracy:", xvAccuracy[xvIndex])
print("Model   :", lModels[xvIndex]) 

# classifier object
# select model with best acc
print("\n*** Classfier Object ***")
model = lModels[xvIndex][1]
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
dfp[clsVars] = le.inverse_transform(dfp[clsVars])
dfp['Predict'] = le.inverse_transform(dfp['Predict'])
print("Done ...")


