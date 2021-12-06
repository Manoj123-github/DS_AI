

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
# utils
import utils

##############################################################
# Read Data 
##############################################################

# read dataset
df = pd.read_csv('C:/Users/Manoj Yadav/Desktop/Part II/ML II/Ensenble algorithms/Iris.csv')


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

# convert string / categoric to numeric
print("\n*** Unique Species - Categoric Alpha to Numeric ***")
print(df[clsVars].unique())
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df[clsVars] = le.fit_transform(df[clsVars])
print(df[clsVars].unique())


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

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required

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
    
# # all categoic variables except clsVars
# # change as required
# colNames = ["varname"]
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
                                test_size=0.2, random_state=707)

# shape
print("\n*** Train & Test Data ***")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("*** Frequency of unique values of the said array ***")
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("*** Frequency of unique values of the said array ***")
print(np.asarray((unique_elements, counts_elements)))


################################
# Classification 
# actual model ... create ... fit ... predict
###############################

# classifier object
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
print("*** Imports ***")
import statistics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
print("Done ...")

# create models
print("*** Create Models ***")
lModels = []
lModels = []
lModels.append(('LogRegr', LogisticRegression(random_state=707)))
lModels.append(('DecTree', DecisionTreeClassifier(random_state=707)))
lModels.append(('KNN-Clf', KNeighborsClassifier()))
for vModel in lModels:
    print(vModel)
print("Done ...")

# blank list to store results
print("\n*** Cross Validation Init ***")
xvModName = []
xvPredict = []
print("Done ...")

# preditc for ensemble
print("\n*** Ensemble Prediction ***")
# iterate through the lModels
for vModelName, oModelObj in lModels:
    # train
    oModelObj.fit(X_train,y_train)
    p_test = oModelObj.predict(X_test)
    cv_acc = accuracy_score(y_test, p_test)*100
    print(vModelName,":  ",cv_acc)
    # update lists for future use
    xvModName.append(vModelName)
    xvPredict.append(p_test)
print("Done ...")

# consolidate prediction result with mean
print("*** Consolidate Prediction ***")
p_test = np.array([])
for i in range(0,len(X_test)):
    #print([xvPredict[0][i], xvPredict[1][i], xvPredict[2][i]])
    p_test = np.append(p_test, statistics.mode([xvPredict[0][i], xvPredict[1][i], xvPredict[2][i]]))
print(p_test)     
print("Done ...")


################################
# Classification 
# Evaluation
###############################

# accuracy
accuracy = accuracy_score(y_test, p_test)*100
print("\n*** Accuracy ***")
print(accuracy)

# confusion matrix - actual
cm = confusion_matrix(y_test, y_test)
print("\n*** Confusion Matrix - Actual ***")
print(cm)

# confusion matrix - predicted
cm = confusion_matrix(y_test, p_test)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_test, p_test)
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

# preditc for ensemble
print("\n*** Ensemble Prediction ***")
# blank list to store results
xvModName = []
xvPredict = []
# iterate through the lModels
for vModelName, oModelObj in lModels:
    # train
    oModelObj.fit(X,y)
    p_pred = oModelObj.predict(X_pred)
    cv_acc = accuracy_score(y_pred, p_pred)*100
    print(vModelName,":  ",cv_acc)
    # update lists for future use
    xvModName.append(vModelName)
    xvPredict.append(y_pred)
print("Done ...")
for i in range(3):
    print(xvModName[i])
    print(xvPredict[i])

# consolidate prediction result with max vote (mode)
print("*** Consolidate Prediction ***")
p_pred = np.array([])
for i in range(0,len(X_pred)):
    p_pred = np.append(p_pred, statistics.mode([xvPredict[0][i], xvPredict[1][i], xvPredict[2][i]]))
print(p_pred)     
print("Done ...")

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
dfp['Predict'] = dfp['Predict'].astype(int)
dfp[clsVars] = le.inverse_transform(dfp[clsVars])
dfp['Predict'] = le.inverse_transform(dfp['Predict'])
print("Done ...")


