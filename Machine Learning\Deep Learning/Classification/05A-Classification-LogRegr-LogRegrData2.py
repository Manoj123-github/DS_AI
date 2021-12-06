

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

# assign spreadsheet filename: file
file = './data/log-regr-data.xlsx'

# load spreadsheet xls
xls = pd.ExcelFile(file)

# print sheet names
print(xls.sheet_names)

# load a sheet into a dataFrame by name
df = xls.parse('MP2')

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
clsVars = "Outcome"
print("\n*** Class Vars ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())


##############################################################
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
print("None ...")

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


##############################################################
# Split Train & Test
##############################################################

# split into data & target
print("\n*** Prepare Data ***")
dfTrain = df.sample(frac=0.8, random_state=707)
dfTest = df.drop(dfTrain.index)
print("Train Count:",len(dfTrain.index))
print("Test Count :",len(dfTest.index))

# split into data & target
print("\n*** Prepare Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(clsVars)
print(allCols)


################################
# Classification 
# actual model ... create ... fit ... predict
###############################

# select based on algo
# uncomment what is required and comment the rest
# only one algo should be selected
# imports
print("\n*** Imports ***")
from statsmodels.api import Logit
from statsmodels.api import add_constant
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print("Done ...")

print("\n*** Recreate Train Data ***")
dfX_train = dfTrain[allCols]
y_train = dfTrain[clsVars].values
print("Done ...")
                      
# model object
print("\n*** Model ***")
# add intercept manually
dfX_train_const = add_constant(dfX_train)
# build model and fit training data
model = Logit(y_train, dfX_train_const).fit()
# print the model summary
print(model.summary()) 
print("Done ...")

################################
# Classification  - Predict Train
# evaluate : Accuracy & Confusion Metrics
###############################

# Probability Distribution for train data
prob_train = model.predict(dfX_train_const)
# sort the prob dist for visualization
sorted_train = sorted(prob_train.values)
index_train = np.arange(len(sorted_train))

# plot it
plt.figure()
sns.regplot(x=index_train, y=sorted_train, color='b', fit_reg=False, scatter_kws={"s": 5})
plt.title('Train Data: Probability Distribution')
plt.xlabel('(sorted by output value)')
plt.ylabel('Probability of Logit function')
plt.show() 

# evaluate
threshold = 0.5
p_train = (prob_train > threshold).astype(np.int8).values
# actual
print("Actual")
print(y_train)
# predicted
print("Predicted")
print(p_train)
print("Done ...")

# accuracy
accuracy = accuracy_score(y_train, p_train)*100
print("\n*** Accuracy ***")
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
cm = confusion_matrix(y_train, y_train)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix
cm = confusion_matrix(y_train, p_train)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_train,p_train)
print(cr)


################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

print("\n*** Recreate Test Data ***")
dfX_test = dfTest[allCols]
y_test = dfTest[clsVars].values
print("Done ...")

# model object
print("\n*** Test Data Predict ***")
# add intercept manually
dfX_test_const = add_constant(dfX_test)
# Probability Distribution for Training data
prob_test = model.predict(dfX_test_const)
threshold = 0.5
p_test = (prob_test > threshold).astype(np.int8).values
# actual
print("Actual")
print(y_test)
# predicted
print("Predicted")
print(p_test)
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
# Create model Object from whole data
# Read .prd file
# Predict Species
# Confusion matrix with data in .prd file
###############################

# classifier object
print("\n*** Classfier Object ***")
dfX = df[allCols]
y = df[clsVars].values                   
# model object
# add intercept manually
dfX_const = add_constant(dfX)
# build model and fit training data
model = Logit(y, dfX_const).fit()
# print the model summary
print(model.summary()) 
print("Done ...")

# read dataset
print("\n*** Read Data For Prediction ***")
data = {'AgeInMonths': [50, 20],
        'Shifts/Week': [6, 5]}
dfp = pd.DataFrame(data)
print("Done ...")

# change as required ... same transformtion as done for main data
print("\n*** Data For Prediction - Transform***")
print("None ...")

# split into data & outcome
print("\n*** Data For Prediction - X & y Split ***")
allCols = dfp.columns.tolist()
print(allCols)
dfX_pred = dfp[allCols]
dfX_pred_const = add_constant(dfX_pred)
print(dfX_pred)
#print(y_pred)

# predict from model
print("\n*** Prediction ***")
prob_pred = model.predict(dfX_pred_const)
threshold = 0.5
p_pred = (prob_pred > threshold).astype(np.int8).values
# actual
print("Actual")
print("N/A")
# predicted
print("Predicted")
print(p_pred)

# update data frame
print("\n*** Update Predict Data ***")
dfp['Predict'] = p_pred
print(dfp)
print("Done ...")
