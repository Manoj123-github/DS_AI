# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 00:18:39 2021
@filename: LinearRegression
@dataset: lr-data
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

# assign spreadsheet filename: file
file = './data/lr-data.xlsx'

# load spreadsheet xls
xls = pd.ExcelFile(file)

# print sheet names
print(xls.sheet_names)

# load a sheet into a dataFrame by name
df = xls.parse('Ex1')


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
depVars = "Revenue"
print("\n*** Dep Vars ***")
print(depVars)


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

# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.2f}'.format
print(df.corr())

# handle multi colinearity if required


##############################################################
# Visual Data Analytics
# https://seaborn.pydata.org/introduction.html#:~:text=Seaborn%20is%20a%20library%20for,explore%20and%20understand%20your%20data.
##############################################################

# check relation with corelation - heatmap
print("\n*** Heat Map ***")
plt.figure(figsize=(4,4))
ax = sns.heatmap(df.corr(), annot=True, cmap="PiYG")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()

# boxplot
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)

# histograms
# https://www.qimacros.com/histogram-excel/how-to-determine-histogram-bin-interval/
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

# scatter plot - hits & rev - spl case
print('\n*** Scatterplot ***')
plt.figure()
sns.regplot(x='Revenue', y='Hits', data=df, color= 'b', scatter_kws={"s": 10})
plt.title('Hits v/s Revenue')
plt.ylabel('Hits')
plt.xlabel('Revenue')
# good practice
plt.show()

##############################################################
# Model Creation & Fitting And Prediction for Feature 
##############################################################

# all cols except dep var
print("\n*** Regression Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(depVars)
print(allCols)

# regression summary for feature
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(df[allCols])
y = df[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# now create linear regression model
print("\n*** Regression Model ***")
X = df[allCols].values.reshape(-1,1)
y = df[depVars].values
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(model)
model.fit(X,y)

# predict
p = model.predict(X)
df['predict'] = p

##############################################################
# Model Evaluation
##############################################################

# visualize 
print("\n*** Scatter Plot ***")
plt.figure()
sns.regplot(data=df, x=depVars, y='predict', color='b', scatter_kws={"s": 10})
plt.show()

# mae 
print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(df[depVars], df['predict'])
print(mae)

# mse
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df[depVars], df['predict'])
print(mse)
   
# rmse 
# https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# check mean
print('\n*** Mean ***')
print(df[depVars].mean())
print(df['predict'].mean())

### accuracy
# MAPE(Mean Absolute Percentage Error)
# https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
# mape = np.mean(np.abs(df['predict'] - df[depVars])/np.abs(df[depVars]))
# acc = 1 - mape

# scatter index (SI) is defined to judge whether RMSE is good or not. 
# SI=RMSE/measured data mean. 
# If SI is less than one, your estimations are acceptable.
# closer to zero the better
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
print('\n*** Scatter Index ***')
si = rmse/df[depVars].mean()
print(si)

# predict for rooms from 200 to 500 step 100
prd_X = np.array([[200],[300],[400],[500]])
print(prd_X.flatten())

prd_p = model.predict(prd_X)
print(prd_p)



    
