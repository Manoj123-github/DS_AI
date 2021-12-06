# -*- coding: utf-8 -*-
"""
Created on Thu May 13 23:30:45 2021

@author: Manoj Yadav
"""

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
# seaborn
import seaborn as sns
# utils
import utils

##############################################################
# Read Data 
##############################################################

# read dataset
df = pd.read_csv('C:/Py-ML/Assignment/gems-reg.csv')

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
#print(df.describe())
print(df.describe(include=np.number))
#print(df.describe(include=np.object))

# head
print("\n*** Head ***")
print(df.head())


##############################################################
# Dependent Variable 
##############################################################

# store dep variable  
# change as required
depVars = "price"
print("\n*** Dep Vars ***")
print(depVars)


##############################################################
# Data Transformation
##############################################################




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

# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.3f}'.format
print(df.corr())

# handle multi colinearity if required


##############################################################
# Visual Data Analytics
##############################################################

# check relation with corelation - heatmap
print("\n*** Heat Map ***")
plt.figure(figsize=(8,8))
ax = sns.heatmap(df.corr(), annot=True, cmap="PiYG")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
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
print('\n*** Histograms ***')
colNames = df.columns.tolist()
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# scatterplots
# plot Sscatterplot
print('\n*** Scatterplot ***')
colNames = df.columns.tolist()
colNames.remove(depVars)
print(colName)
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.regplot(data=df, x=depVars, y=colName, color= 'b', scatter_kws={"s": 5})
    plt.title(depVars + ' v/s ' + colName)
    plt.show()

# class count plot
# change as required
colNames = ["feature1","feature2", "feature3"]
print("\n*** Distribution Plot ***")
for colName in colNames:
    plt.figure()
    sns.countplot(df[colName],label="Count")
    plt.title(colName)
    plt.show()


################################
# Split Train & Test
###############################

# split into data & target
print("\n*** Prepare Data ***")
dftrain = df.sample(frac=0.8, random_state=707)
dftest=df.drop(dftrain.index)
print("Train Count:",len(dftrain.index))
print("Test Count :",len(dftest.index))

##############################################################
# Model Creation & Fitting 
##############################################################

# all cols except dep var 
print("\n*** Regression Data For Train ***")
allCols = dftrain.columns.tolist()
print(allCols)
allCols.remove(depVars)
print(allCols)

# regression summary for feature
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(dftrain[allCols])
y = dftrain[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# remove columns with p-value > 0.05

allCols.remove('feature3')
print(allCols)

# regression summary for feature
print("\n*** Regression Summary Again ***")
X = sm.add_constant(dftrain[allCols])
y = dftrain[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# now create linear regression model
print("\n*** Regression Model ***")
X = dftrain[allCols].values
y = dftrain[depVars].values
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
print(model)
print("Done ...")

##############################################################
# predict with train data 
##############################################################

# predict
print("\n*** Predict - Train Data ***")
p = model.predict(X)
dftrain['predict'] = p
print("Done ...")

##############################################################
# Model Evaluation - Train Data
##############################################################

# visualize 
print("\n*** Scatter Plot ***")
plt.figure()
sns.regplot(data=dftrain, x=depVars, y='predict', color='b', scatter_kws={"s": 5})
plt.show()

# mae 
print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(dftrain[depVars], dftrain['predict'])
print(mae)

# mse 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(dftrain[depVars], dftrain['predict'])
print(mse)
   
# rmse 
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# # check mean
print('\n*** Mean ***')
print(dftrain[depVars].mean())
print(dftrain['predict'].mean())

# # scatter index (SI) is defined to judge whether RMSE is good or not. 
# # SI=RMSE/measured data mean. 
# # If SI is less than one, your estimations are acceptable.
print('\n*** Scatter Index ***')
si = rmse/dftrain[depVars].mean()
print(si)


##############################################################
# confirm with test data 
##############################################################

# all cols except dep var 
print("\n*** Regression Data For Test ***")
print(allCols)
# split
X = dftest[allCols].values
y = dftest[depVars].values
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

# predict
print("\n*** Predict - Train Data ***")
p = model.predict(X)
dftest['predict'] = p
print("Done ...")

##############################################################
# Model Evaluation - Test Data
##############################################################

# visualize 
print("\n*** Scatter Plot ***")
plt.figure()
sns.regplot(data=dftest, x=depVars, y='predict', color='b', scatter_kws={"s": 5})
plt.show()

# mae 
print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(dftest[depVars], dftest['predict'])
print(mae)

# mse 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(dftest[depVars], dftest['predict'])
print(mse)
   
# rmse 
# RMSE measures the error.  How good is an error depends on the amplitude of your data. 
# RMSE should be less 10% for mean(depVars)
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# check mean
print('\n*** Mean ***')
print(dftest[depVars].mean())
print(dftest['predict'].mean())
 
# # scatter index
# # scatter index less than 1; the predictions are decent
print('\n*** Scatter Index ***')
si = rmse/dftest[depVars].mean()
print(si)


##############################################################
# predict from new data 
##############################################################

# create model from full dataset
print(allCols)

# regression summary for feature
print("\n*** Regression Summary Again ***")
X = sm.add_constant(df[allCols])
y = df[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# now create linear regression model
print("\n*** Regression Model ***")
X = df[allCols].values
y = df[depVars].values
model = LinearRegression()
model.fit(X,y)
print(model)

# read dataset
dfp = pd.read_csv('C:/Py-ML/Assignment/gems-reg-prd.csv')

print("\n*** Structure ***")
print(dfp.info())


# split X & y
prd_X = dfp[allCols].values
prd_y = dfp[depVars].values

# predict
prd_p = model.predict(prd_X)
dfp['predict'] = prd_p

# # visualize 
print("\n*** Scatter Plot ***")
plt.figure()
sns.regplot(data=dfp, x=depVars, y='predict', color= 'b')
plt.show()

# # mae 
 print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(dfp[depVars], dfp['predict'])
print(mae)

# # mse
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(dfp[depVars], dfp['predict'])
print(mse)

# # rmse
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# # check mean
print('\n*** Mean ***')
print(dfp[depVars].mean())
print(dfp['predict'].mean())

# # scatter index
# # scatter index less than 1; the predictions are good
print('\n*** Scatter Index ***')
si = rmse/dfp[depVars].mean()
print(si)


    