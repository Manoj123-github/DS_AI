

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
df = pd.read_csv('C:/Users/Manoj Yadav/Desktop/Part II/ML II/Ensenble algorithms/medins-masked.csv')


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
clsVars = "label"
print("\n*** Class Variable ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# get unique Class names
print("\n*** Unique Class - Categoric Numeric ***")
print(df[clsVars].unique())

# not required as class var is already numeric


##############################################################
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
df = df.drop('col_0', axis=1)
print("Done ...")

# convert alpha categoric to numeric categoric
# change as required

# col_2
from sklearn import preprocessing
leCol2 = preprocessing.LabelEncoder()
print(df['col_2'].unique())
df['col_2'] = leCol2.fit_transform(df['col_2'])
print(df['col_2'].unique())

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# # handle outlier
# print('\n*** Handle Outliers ***')
# colNames = []
# for colName in colNames:
#       colType =  df[colName].dtype  
#       df[colName] = utils.HandleOutliers(df[colName])
#       if df[colName].isnull().sum() > 0:
#           df[colName] = df[colName].astype(np.float64)
#       else:
#           df[colName] = df[colName].astype(colType)    
# print("Done ...")


# # check outlier count
# print('\n*** Outlier Count ***')
# print(utils.OutlierCount(df))

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

# # check variance
# print('\n*** Variance In Columns ***')
# print(df.var())

# # check std dev 
# print('\n*** StdDev In Columns ***')
# print(df.std())

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros
df['col_2'] = np.where(df['col_2']==0,None, df['col_2'])
df['col_3'] = np.where(df['col_3']==0,None, df['col_3'])
df['col_7'] = np.where(df['col_7']==0,None, df['col_7'])

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls
df['col_1'] = df['col_1'].fillna(df['col_1'].mean())
df['col_2'] = df['col_2'].fillna(df['col_2'].mean())
df['col_3'] = df['col_3'].fillna(df['col_3'].mean())
df['col_7'] = df['col_7'].fillna(df['col_7'].mean())
df['col_8'] = df['col_8'].fillna(df['col_8'].mean())

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# # feature selection
# print("\n*** Feature Scores - XTC ***")
# print(utils.getFeatureScoresXTC(df, clsVars))

# print("\n*** Feature Scores - SKC ***")
# print(utils.getFeatureScoresSKB(df, clsVars))

# drop cols
# change as required
# print("\n*** Drop Cols ***")
# print("None ...")

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
# Classification - set X & y
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

# # over sampling
# print("\n*** Over Sampling Process ***")
# X, y = utils.getOverSamplerData(X, y)
# print("Done ...")

# # counts
# print("\n*** Counts ***")
# unique_elements, counts_elements = np.unique(y, return_counts=True)
# print(np.asarray((unique_elements, counts_elements)))

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))


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
maxXVIndx = xvAccuracy.index(max(xvAccuracy))
print("Index      : ",maxXVIndx)
print("Model Name : ",xvModNames[maxXVIndx])
print("XVAccuracy : ",xvAccuracy[maxXVIndx])
print("XVStdDev   : ",xvSDScores[maxXVIndx])
print("Model      : ",lModels[maxXVIndx])


################################
# Classification - Split Train & Test
###############################

# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
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
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))


################################
# Finding best parameters for our model
###############################
from sklearn.model_selection import GridSearchCV
print("\n*** Grid Search XV For " + xvModNames[maxXVIndx] + " ***")
blnGridSearch = True
# SVC model
if xvModNames[maxXVIndx] == 'SVM-Clf':
    param = { 
        'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
        'kernel':['linear', 'rbf'],
        'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
    }
    gsxv = GridSearchCV(SVC(), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# random forest model
if xvModNames[maxXVIndx] == 'RndFrst':
    param = { 
        'n_estimators': [100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth'   : [4,5,6,7,8],
        'criterion'   : ['gini', 'entropy']
    }
    gsxv = GridSearchCV(RandomForestClassifier(random_state=707), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# knn classifir
if xvModNames[maxXVIndx] == 'KNN-Clf':
    param = { 
        'leaf_size' : list(range(1,50)), 
        'n_neighbors' : list(range(1,30)), 
        'p' : [1,2]
    }
    gsxv = GridSearchCV(KNeighborsClassifier(), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# logistic regression
if xvModNames[maxXVIndx] == 'LogRegr':
    param = { 
        'C':[0.001,.009,0.01,.09,1,5,10,15,20,25],
        'penalty': ['l1', 'l2',13,14,15]        
    }
    gsxv = GridSearchCV(LogisticRegression(), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# decision tree
if xvModNames[maxXVIndx] == 'DecTree':
    param = { 
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth'   : [4,5,6,7,8],
        'criterion'   : ['gini', 'entropy']
    }
    gsxv = GridSearchCV(DecisionTreeClassifier(), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# naive bayes
if xvModNames[maxXVIndx] == 'GNBayes':
    param = { 
        'vec__max_df': (0.5, 0.625, 0.75, 0.875, 1.0),  
        'vec__max_features': (None, 5000, 10000, 20000),  
        'vec__min_df': (1, 5, 10, 20, 50),  
    }        
    gsxv = GridSearchCV(GaussianNB(), param_grid=param, scoring='accuracy', verbose=10, cv=5)
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
# show verbose    
gsxv.fit(X_train, y_train)
bestParams = gsxv.best_params_
print("\n*** Grid Search XV For " + xvModNames[maxXVIndx] + " ***")
print(bestParams)
print(type(bestParams))


################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

print("\n*** Accuracy & Models ***")
print("Accuracy:", xvAccuracy[maxXVIndx])
print("Model   :", xvModNames[maxXVIndx]) 

# classifier object
print("\n*** Classfier Object ***")
if blnGridSearch==False:
    print(blnGridSearch)
    model = lModels[maxXVIndx][1]
else:
    if xvModNames[maxXVIndx] == 'SVM-Clf':
        model = SVC(C=bestParams['C'], kernel=bestParams['kernal'], gamma=bestParams['gamma'])
    if xvModNames[maxXVIndx] == 'RndFrst':
        model = RandomForestClassifier(criterion=bestParams['criterion'], max_depth=bestParams['max_depth'], max_features=bestParams['max_features'], n_estimators=bestParams['n_estimators'], random_state=707)
    if xvModNames[maxXVIndx] == 'KNN-Clf':
        model = KNeighborsClassifier(leaf_size=bestParams['leaf_size'], n_neighbors=bestParams['n_neighbors'], p=bestParams['p'])
    if xvModNames[maxXVIndx] == 'LogRegr':
        model = LogisticRegression(C=bestParams['C'], penalty=bestParams['penalty'])
    if xvModNames[maxXVIndx] == 'DecTree':
        model = DecisionTreeClassifier(max_features=bestParams['max_features'],max_depth=bestParams['max_depth'],criterion=bestParams['criterion'])
    if xvModNames[maxXVIndx] == 'GNBayes':
        model = GaussianNB(vec__max_df=bestParams['vec__max_df'],vec__max_features=bestParams['vec__max_features'],vec__min_df=bestParams['vec__min_df'])
print(model)


################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

print("\n*** Accuracy & Models ***")
print("Accuracy:", xvAccuracy[maxXVIndx])
print("Model   :", model) 

# classifier model already selected ... now train
# fit the model
print("\n*** Model Train ***")
model.fit(X_train,y_train)
print("Done ...")

# predict test
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
dfp = pd.read_csv('C:/Users/Manoj Yadav/Desktop/Part II/ML II/Ensenble algorithms/medins-masked-prd.csv')
print(dfp.info())

# not required
print("\n*** Data For Prediction - Drop Cols***")
print("N/A ...")

# convert string / categoric to numeric
print("\n*** Data For Prediction - Class Vars ***")
print(dfp[clsVars].unique())

# change as required ... same transformtion as done for main data
print("\n*** Data For Prediction - Transform***")
dfp = dfp.drop('col_0', axis=1)
print(dfp['col_2'].unique())
dfp['col_2'] = leCol2.transform(dfp['col_2'])
print(dfp['col_2'].unique())

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

# model
print("\n*** Accuracy & Models ***")
print("Accuracy:", xvAccuracy[maxXVIndx])
print("Model   :", model) 

# train the model
print("\n*** Model Train ***")
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
