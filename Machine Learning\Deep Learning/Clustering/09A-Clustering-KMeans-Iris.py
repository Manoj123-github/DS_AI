
# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
import seaborn as sns
import utils


##############################################################
# Read Data 
##############################################################

# read dataset
print("\n*** Read Data ***")
df = pd.read_csv('C:/Users/Manoj Yadav/Desktop/ML/Clustering/Iris.csv')
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

# shape
print("\n*** Shape ***")
print(df.shape)


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
# drop cols which contain identifiers, nominals, descriptions
# change as required
print("\n*** Drop Cols ***")
dfId = df['Id']     # store Id in dfID to recreate dataframe later
df = df.drop('Id', axis=1)
print("Done ...")

# transformations
# change as required
print("\n*** Transformations ***")
print("None ...")

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
# Visual Data Analytics
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

# plot sepal
plt.figure()
sns.lmplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species', fit_reg=False)
plt.title('Sepal')
plt.show()

# plot petal
plt.figure()
sns.lmplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species', fit_reg=False)
plt.title('Petal')
plt.show()

# # all categoic variables except clsVars
# # change as required
# colNames = ["colName"]
# print("\n*** Distribution Plot ***")
# for colName in colNames:
#     plt.figure()
#     sns.countplot(df[colName],label="Count")
#     plt.title(colName)
#     plt.show()


################################
# Prepare Data
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
# KMeans Clustering 
###############################

# imports
from sklearn.cluster import KMeans

# how to decide on the clusters
# within cluster sum of squares errors - wcsse
# elbow method ... iterations should be more than 10
print("\n*** Compute WCSSE ***")
vIters = 11
lWcsse = []
for i in range(1, vIters):
    kmcModel = KMeans(n_clusters=i)
    kmcModel.fit(X)
    lWcsse.append(kmcModel.inertia_)
for vWcsse in lWcsse:
    print(vWcsse)

# plotting the results onto a line graph, allowing us to observe 'The elbow'
print("\n*** Plot WCSSE ***")
plt.figure()
plt.plot(range(1, vIters), lWcsse)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSSE') #within cluster sum of squares error
plt.show()

# programatically
#!pip install kneed
print("\n*** Find Best K ***")
import kneed
kl = kneed.KneeLocator(range(1, vIters), lWcsse, curve="convex", direction="decreasing")
vBestK = kl.elbow
print(vBestK)

# you can clearly see why it is called 'The elbow method' from the above graph, 
# the optimum clusters is where the elbow occurs. This is when the within cluster 
# sum of squares (WCSS) doesn't decrease significantly with every iteration. 
# Now that we have the optimum amount of clusters, we can move on to applying 
# K-means clustering to the Iris dataset

# k means cluster model
print("\n*** Model Create & Train ***")
model = KMeans(n_clusters=vBestK, random_state=707)
model.fit(X)

# result
print("\n*** Model Results ***")
print(model.labels_)
df['Predict'] = model.labels_
# recreate dataframe 
df = pd.concat([dfId, df], axis=1)

# compare
print("\n*** Actual ***")
print(df[clsVars].tolist())
print("")
print("*** Predicted ***")
print(df['Predict'].tolist())

## interchange as required ...could be different for each group
df['Predict'] = df['Predict'].map({1:0, 0:1, 2:2})

print("\n*** Actual ***")
print(df[clsVars].tolist())
print("")
print("*** Predicted ***")
print(df['Predict'].tolist())

# imports
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import v_measure_score
# confusion matrix
print("\n*** Confusion Matrix - Actual ***")
cm = confusion_matrix(df[clsVars], df[clsVars])
print(cm)
print("\n*** Confusion Matrix - Clustered ***")
cm = confusion_matrix(df[clsVars], df['Predict'])
print(cm)
# accuracy 
print("\n*** Accuracy ***")
ac = accuracy_score(df['Species'], df['Predict'])*100
print(ac)
# v-measure score
# checks clusters on two counts
# how homogeneous a cluster is 
# how complete cluster is
# https://towardsdatascience.com/v-measure-an-homogeneous-and-complete-clustering-ab5b1823d0ad
print("\n*** V-Score ***")
vm = v_measure_score(df['Species'], df['Predict'])
print(vm)