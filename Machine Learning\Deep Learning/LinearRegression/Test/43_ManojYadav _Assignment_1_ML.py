# -*- coding: utf-8 -*-
"""
Created on Sat May  8 08:15:28 2021

@author: Manoj Yadav
"""


# hides all warnings
import warnings
warnings.filterwarnings('ignore')


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

# Read Data 
##############################################################

file = 'C:/Users/Manoj Yadav/Desktop/slr-cyrogenic-flows.xlsx'


# load spreadsheet xls
xlsx = pd.ExcelFile(file)

# print sheet names
print(xlsx.sheet_names)

# load a sheet into a dataFrame by name
df = xlsx.parse('Sheet1')

# Exploratory Data Analytics
##############################################################
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

# Dependent Variable 
##############################################################

# store dep variable  
# change as required
depVars = "y"
print("\n*** Dep Vars ***")
print(depVars)

# Data Transformation
##############################################################


# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers if required
#cols=['y']
#Q1=df[cols].quantile(0.25)
#Q3=df[cols].quantile(0.75)
#IQR=Q3-Q1

#df=df[~((df[cols]<(Q1-1.5*IQR))|(df[cols]>(Q3+1.5*IQR))).any(axis=1)]

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# check mean
print('\n*** Mean In Columns ***')
print(df.mean())


# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 


# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.2f}'.format
print(df.corr())


##############################################################
# Visual Data Analytics

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
sns.regplot(x='x', y='y', data=df, color= 'b', scatter_kws={"s": 10})
plt.title('x v/s y')
plt.ylabel('y')
plt.xlabel('x')
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
sns.regplot(data=df, y=depVars, x='predict', color='b', scatter_kws={"s": 10})
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
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# check mean
print('\n*** Mean ***')
print(df[depVars].mean())
print(df['predict'].mean())

### accuracy
# MAPE(Mean Absolute Percentage Error)
# mape = np.mean(np.abs(df['predict'] - df[depVars])/np.abs(df[depVars]))
# acc = 1 - mape

# scatter index (SI) is defined to judge whether RMSE is good or not. 
# SI=RMSE/measured data mean. 
# If SI is less than one, your estimations are acceptable.
# closer to zero the better
print('\n*** Scatter Index ***')
si = rmse/df[depVars].mean()
print(si)

# predict for rooms from 200 to 500 step 100
prd_X = np.array([[60],[70],[80],[90],[100],[110],[120]])
print(prd_X.flatten())

prd_p = model.predict(prd_X)
print(prd_p)
