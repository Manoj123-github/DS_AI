

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

# col names
colNames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# read dataset
df = pd.read_csv('./data/boston-housing.dat', header=None, delimiter=r"\s+", names=colNames)


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
depVars = 'MEDV'
print("\n*** Dep Vars ***")
print(depVars)


##############################################################
# Data Transformation
##############################################################

# drop cols which contain identifiers, nominals, descriptions
# change as required
print("\n*** Drop Cols ***")
#df = df.drop('id', axis=1)
print("None ...")

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
# data.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
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
colNames = ["CHAS","RAD"]
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
dfTrain = df.sample(frac=0.8, random_state=707)
dfTest = df.drop(dfTrain.index)
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

# regression summary for feature
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
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
allCols.remove('INDUS')
allCols.remove('AGE')
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


###############################
# multi algos
###############################
# Ridge Regression:
# Ridge Regression added a term in ordinary least square error function that 
# regularizes the value of coefficients of variables. This term is the sum of 
# squares of coefficient multiplied by the parameter The motive of adding this 
# term is to penalize the variable corresponding to that coefficient not very 
# much correlated to the target variable. This term is called L2 regularization.
#
# Lasso Regression:
# Lasso Regression is similar to Ridge regression except here we add Mean 
# Absolute value of coefficients in place of mean square value. Unlike Ridge 
# Regression, Lasso regression can completely eliminate the variable by 
# reducing its coefficient value to 0. The new term we added to Ordinary 
# Least Square(OLS) is called L1 Regularization.
#
# Elastic Net :
# In elastic Net Regularization we added the both terms of L1 and L2 to get 
# the final loss function. 

print("\n*** Regression Model ***")

# imports 
# change as required
# normal linear regression
from sklearn.linear_model import LinearRegression 
# ridge regression 
# from sklearn.linear_model import Ridge 
# Lasso regression 
# from sklearn.linear_model import Lasso 
# import model 
# from sklearn.linear_model import ElasticNet 

# model names
# change as required
vModelName = "Normal" 
#vModelName = "Ridge" 
#vModelName = "Lasso" 
#vModelName = "Elasti" 
 
# create model object
# change as required
model = LinearRegression() 
#model = Ridge(alpha = 10) 
#model = Lasso(alpha = 1)
#model = ElasticNet(alpha = 1)
print(model)
print("Done ...")

# fit / train the model
print("\n*** Regression Model - Fit / Train ***")
model.fit(X_train, y_train) 
print("Done ...")

##############################################################
# predict with train data 
##############################################################

# predict
print("\n*** Predict - Train Data ***")
p_train = model.predict(X_train)
dfTrain['predict'] = p_train
print("Done ...")

##############################################################
# Model Evaluation - Train Data
##############################################################

# visualize 
print("\n*** Scatter Plot ***")
plt.figure()
#sns.regplot(data=dfTrain, x=depVars, y='predict', color='b', scatter_kws={"s": 5})
sns.regplot(x=y_train, y=p_train, color='b', scatter_kws={"s": 5})
plt.show()

# R-Square
print('\n*** R-Square ***')
from sklearn.metrics import r2_score
#r2 = r2_score(dfTrain[depVars], dfTrain['predict'])
r2 = r2_score(y_train, p_train)
print(r2)

# adj r-square  
print('\n*** Adj R-Square ***')
adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))
print(adj_r2)

# mae 
print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_train, p_train)
print(mae)

# mse 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_train, p_train)
print(mse)
   
# rmse 
#https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# check mean
print('\n*** Mean ***')
print(y_train.mean())
print(p_train.mean())

# scatter index (SI) is defined to judge whether RMSE is good or not. 
# SI=RMSE/measured data mean. 
# If SI is less than one, your estimations are acceptable.
print('\n*** Scatter Index ***')
si = rmse/y_train.mean()
print(si)

##############################################################
# confirm with test data 
##############################################################

# test data
print("\n*** Regression Data For Test ***")
print(allCols)
# split
X_test = dfTest[allCols].values
y_test = dfTest[depVars].values
# print
print(X_test.shape)
print(y_test.shape)
print(type(X_test))
print(type(y_test))
print("Done ...")

# predict
print("\n*** Predict - Test Data ***")
p_test = model.predict(X_test)
dfTest['predict'] = p_test
print("Done ...")

##############################################################
# Model Evaluation - Test Data
##############################################################

# visualize 
print("\n*** Scatter Plot ***")
plt.figure()
#sns.regplot(data=dfTest, x=depVars, y='predict', color='b', scatter_kws={"s": 5})
sns.regplot(x=y_test, y=p_test, color='b', scatter_kws={"s": 5})
plt.show()

# R-Square
print('\n*** R-Square ***')
from sklearn.metrics import r2_score
r2 = r2_score(y_train, p_train)
print(r2)

# adj r-square  
print('\n*** Adj R-Square ***')
adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))
print(adj_r2)

# mae 
print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, p_test)
print(mae)

# mse 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, p_test)
print(mse)
   
# rmse 
# RMSE measures the error.  How good is an error depends on the amplitude of your data. 
# RMSE should be less 10% for mean(depVars)
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# check mean
print('\n*** Mean ***')
print(y_test.mean())
print(p_test.mean())
 
# scatter index
# scatter index less than 1; the predictions are decent
print('\n*** Scatter Index ***')
si = rmse/y_test.mean()
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
colNames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dfp = pd.read_csv('./data/boston-housing-prd.csv', header=None, delimiter=r"\s+", names=colNames)

print("\n*** Structure ***")
print(dfp.info())

# drop cols 
# change as required
print("\n*** Drop Cols ***")
# dfp = dfp.drop('id', axis=1)
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
