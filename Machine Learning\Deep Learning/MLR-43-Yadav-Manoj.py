# -*- coding: utf-8 -*-
"""
Created on Sat May 22 14:23:59 2021

@author: Manoj Yadav
"""

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
# seaborn
import seaborn as sns
# utils
import utils

##############################################################
# Read Data 
##############################################################

# read dataset
df = pd.read_csv('C:/Py-ML//california-housing.csv')


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
# Dependent Variable 
##############################################################

# store dep variable  
# change as required
depVars = 'median_house_value'
print("\n*** Dep Vars ***")
print(depVars)


##############################################################
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
df = df.drop(['ser','longitude','latitude','ocean_proximity'] ,axis=1)
print("Done ...")

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))



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
df=df.dropna()

# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.3f}'.format
print(df.corr())
print("Done ...")


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



###############################
# Split Train & Test
###############################

# split into data & target
print("\n*** Prepare Data ***")
dfTrain = df.sample(frac=0.8, random_state=707)
dfTest=df.drop(dfTrain.index)
print("Train Count:",len(dfTrain.index))
print("Test Count :",len(dfTest.index))

##############################################################
# Model Creation & Fitting 
##############################################################

# all cols except dep var 
print("\n*** Regression Data ***")
allCols = dfTrain.columns.tolist()
print(allCols)
allCols.remove(depVars)
print(allCols)
print("Done ...")

# regression summary for feature
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(dfTrain[allCols])
y = dfTrain[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# remove columns with p-value > 0.05
# change as required
print("\n*** Drop Cols ***")
allCols.remove('random_income')
print(allCols)

# regression summary for feature
print("\n*** Regression Summary Again ***")
X = sm.add_constant(dfTrain[allCols])
y = dfTrain[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# train data
print("\n*** Regression Data For Train ***")
X_train = dfTrain[allCols].values
y_train = dfTrain[depVars].values
# print
print(X_train.shape)
print(y_train.shape)
print(type(X_train))
print(type(y_train))
print("Done ...")

# test data
print("\n*** Regression Data For Test ***")
X_test = dfTest[allCols].values
y_test = dfTest[depVars].values
print(X_test.shape)
print(y_test.shape)
print(type(X_test))
print(type(y_test))
print("Done ...")


###############################
# Auto Select Best Regression
###############################

# imports 
print("\n*** Import Regression Libraries ***")
# normal linear regression
from sklearn.linear_model import LinearRegression 
# ridge regression from sklearn library 
from sklearn.linear_model import Ridge 
# import Lasso regression from sklearn library 
from sklearn.linear_model import Lasso 
# import model 
from sklearn.linear_model import ElasticNet 
print("Done ...")
  
# empty lists
print("\n*** Init Empty Lists ***")
lModels = []
lModelAdjR2 = []
lModelRmses = []
lModelScInd = []
lModelCoefs = []
print("Done ...")

# list model name list
print("\n*** Init Models Lists ***")
lModels.append(("LinearRegression", LinearRegression()))
lModels.append(("RidgeRegression ", Ridge(alpha = 10)))
lModels.append(("LassoRegression ", Lasso(alpha = 1)))
lModels.append(("ElasticNet      ", ElasticNet(alpha = 1)))
print("Done ...")

# iterate through the models list
for vModelName, oModelObject in lModels:
    # create model object
    model = oModelObject
    # print model vals
    print("\n*** "+vModelName)
    # fit or train the model
    model.fit(X_train, y_train) 
    # predict train set 
    y_pred = model.predict(X_train)
    dfTrain[vModelName] = y_pred
    # predict test set 
    y_pred = model.predict(X_test)
    dfTest[vModelName] = y_pred
    # r-square  
    from sklearn.metrics import r2_score
    r2 = r2_score(dfTrain[depVars], dfTrain[vModelName])
    print("R-Square:",r2)
    # adj r-square  
    adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
              (X_train.shape[0] - X_train.shape[1] - 1)))
    lModelAdjR2.append(adj_r2)
    print("Adj R-Square:",adj_r2)
    # mae 
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(dfTest[depVars], dfTest[vModelName])
    print("MAE:",mae)
    # mse 
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(dfTest[depVars], dfTest[vModelName])
    print("MSE:",mse)
    # rmse 
    rmse = np.sqrt(mse)
    lModelRmses.append(rmse)
    print("RMSE:",rmse)
    # scatter index
    si = rmse/dfTest[depVars].mean()
    lModelScInd.append(si)
    print("SI:",si)

# print key metrics for each model
msg = "%10s %16s %10s %10s" % ("Model Type", "AdjR2", "RMSE", "SI")
print(msg)
for i in range(0,len(lModels)):
    # print accuracy mean & std
    msg = "%16s %10.3f %10.3f %10.3f" % (lModels[i][0], lModelAdjR2[i], lModelRmses[i], lModelScInd[i])
    print(msg)


# find model with best adj-r2 & print details
print("\n*** Best Model ***")
vBMIndex = lModelAdjR2.index(max(lModelAdjR2))
print("Index       : ",vBMIndex)
print("Model Name  : ",lModels[vBMIndex][0])
print("Adj-R-Sq    : ",lModelAdjR2[vBMIndex])
print("RMSE        : ",lModelRmses[vBMIndex])
print("ScatterIndex: ",lModelScInd[vBMIndex])


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
model = lModels[vBMIndex][1]
model.fit(X,y)
print(model)

# read dataset

dfp = pd.read_csv('C:/Py-ML/california-housing-prd.csv')

print("\n*** Structure ***")
print(dfp.info())

# drop cols 
# change as required
print("\n*** Drop Cols ***")
dfp = dfp.drop(['ser','longitude','latitude'] ,axis=1)
print("None ... ")

# transformation
# change as required
print("\n*** Transformation ***")
print("None ... ")

# split X & y
print("\n*** Split Predict Data ***")
X_pred = dfp[allCols].values
y_pred = dfp[depVars].values
print(X_pred)
print(y_pred)

# predict
print("\n*** Predict Data ***")
p_pred = model.predict(X_pred)
dfp['predict'] = p_pred
print("Done ... ")

# visualize 
print("\n*** Scatter Plot ***")
plt.figure()
sns.regplot(x=y_pred, y=p_pred, color= 'b')
plt.show()

# mae 
print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_pred, p_pred)
print(mae)

# mse
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_pred, p_pred)
print(mse)

# rmse
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# check mean
print('\n*** Mean ***')
print(y_pred.mean())
print(p_pred.mean())

# scatter index
# scatter index less than 1; the predictions are good
print('\n*** Scatter Index ***')
si = rmse/y_pred.mean()
print(si)