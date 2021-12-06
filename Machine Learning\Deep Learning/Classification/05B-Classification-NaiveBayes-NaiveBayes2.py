
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
plt.rcParams['figure.figsize'] = (10, 5)
# sns
import seaborn as sns
# util
import utils


##############################################################
# Read Data 
##############################################################

# read dataset
df = pd.read_csv('./data/naive-bayes.csv')


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
clsVars = "Play"
print("\n*** Class Vars ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# get unique Species names
print("\n*** Unique Play - Categoric Alpha***")
lnLabels = df[clsVars].unique()
print(lnLabels)

# convert string / categoric to numeric
print("\n*** Unique Play - Categoric Alpha to Numeric ***")
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
#df = df.drop('Id', axis=1)
print("None ...")

# convert string to numeric
print("\n*** Transformations ***")
print("Outlook")
print(df['Outlook'].unique())
df['Outlook'] = df['Outlook'].map({"Sunny": 2, "Cloudy": 1, "Rain": 0})
print(df['Outlook'].unique())
print("Temperature")
print(df['Temp'].unique())
df['Temp'] = df['Temp'].map({"Hot": 2, "Mild": 1, "Cool": 0})
print(df['Temp'].unique())
print("Humidity")
print(df['Humid'].unique())
df['Humid'] = df['Humid'].map({"High": 2, "Normal": 1, "Low": 0})
print(df['Humid'].unique())
print("Wind")
print(df['Wind'].unique())
df['Wind'] = df['Wind'].map({"Strong": 2, "Weak": 1, "Poor": 0})
print(df['Humid'].unique())
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
    
# all categoic variables except clsVars
# # change as required
colNames = df.columns.tolist()
colNames.remove(clsVars)
for colName in colNames:
     plt.figure()
     sns.countplot(df[colName],label="Count")
     plt.title(colName)
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

# imports
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# classifier object
print("\n*** Classfier ***")
# create model ... empty model
model = GaussianNB()
print(model)
# fit the model
model.fit(X_train, y_train)                # create model ... fit data


################################
# Classification  - Predict Train
# evaluate : Accuracy & Confusion Metrics
###############################

# classifier object
print("\n*** Predict Train ***")
# predicting the Test set results
p_train = model.predict(X_train)            # use model ... predict
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
# X-axis Predicted | Y-axis Actual
cm = confusion_matrix(y_train, p_train)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_train,p_train)
print(cr)

# make dftrain
# only for demo
# not to be done in production
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

# make dftest
# only for show
# not to be done in production
print("\n*** Recreate Test ***")
dfTest =  pd.DataFrame(data = X_test)
dfTest.columns = allCols
dfTest[clsVars] = y_test
dfTest['Predict'] = p_test
dfTest[clsVars] = le.inverse_transform(dfTest[clsVars])
dfTest['Predict'] = le.inverse_transform(dfTest['Predict'])
print("Done ...")


################################
# Final Prediction
# Create model Object from whole data
# Read .prd file
# Predict Species
# Confusion matrix with data in .prd file
###############################

# classifier object
print("\n*** Classfier Object ***")
model = GaussianNB()
print(model)
# fit the model
model.fit(X,y)
print("Done ...")

# read dataset
print("\n*** ata For Prediction ***")
data = {'Outlook': ['Cloudy','Sunny'],
        'Temp'   : ['Mild', 'Hot'], 
        'Humid'  : ['High','High'],
        'Wind'   : ['Weak','Strong']}
dfp = pd.DataFrame(data)
print(dfp)

# change as required ... same transformtion as done for main data
print("\n*** Data For Prediction - Transform***")
#dfp = dfp.drop('Id', axis=1)
dfp['Outlook'] = dfp['Outlook'].map({"Sunny": 2, "Cloudy": 1, "Rain": 0})
dfp['Temp']    = dfp['Temp'].map({"Hot": 2, "Mild": 1, "Cool": 0})
dfp['Humid']   = dfp['Humid'].map({"High": 2, "Normal": 1, "Low": 0})
dfp['Wind']    = dfp['Wind'].map({"Strong": 2, "Weak": 1, "Poor": 0})
print("Done ...")

# split into data & outcome
print("\n*** Data For Prediction - X & y Split ***")
allCols = dfp.columns.tolist()
#print(allCols)
#allCols.remove(clsVars)
#print(allCols)
X_pred = dfp[allCols].values
#y_pred = dfp[clsVars].values
print(X_pred)
#print(y_pred)

# predict from model
print("\n*** Prediction ***")
p_pred = model.predict(X_pred)
# actual
#print("Actual")
#print(y_pred)
# predicted
print("Predicted")
print(p_pred)

# update data frame
print("\n*** Update Predict Data ***")
dfp['Predict'] = p_pred
#dfp[clsVars] = le.inverse_transform(dfp[clsVars])
dfp['Predict'] = le.inverse_transform(dfp['Predict'])
print("Done ...")
