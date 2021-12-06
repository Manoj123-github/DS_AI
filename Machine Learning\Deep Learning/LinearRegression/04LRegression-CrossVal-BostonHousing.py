

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

# read
print("\n*** Read Data ***")
# col names
colNames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# read dataset
df = pd.read_csv('C:/Users/Manoj Yadav/Desktop/Part II/ML II/Regression/boston-housing.dat', header=None, delimiter=r"\s+", names=colNames)
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

# drop cols
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
pd.options.display.float_format = '{:,.2f}'.format
dfc = df.corr()
print("Done ...")

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
# Regression -  set X & y
###############################

# all cols except dep var 
print("\n*** Regression Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(depVars)
print(allCols)
print("Done ...")

# train data
print("\n*** Split Regression Data In X & y ***")
X = df[allCols].values
y = df[depVars].values

##############################################################
# feature reduction 
##############################################################

# regression summary for feature
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(df[allCols])
y = df[depVars]
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
X = sm.add_constant(df[allCols])
y = df[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# full data
print("\n*** Regression Data ***")
X = df[allCols].values
y = df[depVars].values
# print
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))
print("Done ...")

################################
# Regression - init models
###############################

# https://www.kaggle.com/jnikhilsai/cross-validation-with-linear-regression

# imports 
print("\n*** Import Regression Libraries ***")
# normal linear regression
from sklearn.linear_model import LinearRegression 
# ridge regression from sklearn library 
from sklearn.linear_model import Ridge 
# import Lasso regression from sklearn library 
from sklearn.linear_model import Lasso 
# import ElasticNet model 
from sklearn.linear_model import ElasticNet 
# import RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor 
# decision tree regressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor
#from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
# svm regressor
from sklearn.svm import SVR
print("Done ...")

# list model name list
print("\n*** Init Models Lists ***")
lModels = []
lModels.append(("LinearRegression  ", LinearRegression()))
lModels.append(("RidgeRegression   ", Ridge(alpha = 10)))
lModels.append(("LassoRegression   ", Lasso(alpha = 1)))
lModels.append(("ElasticNet        ", ElasticNet(alpha = 1)))
lModels.append(("Random Forest     ", RandomForestRegressor(random_state = 707)))
lModels.append(("SVM Regressor     ", SVR(C=1.0, epsilon=0.2)))
lModels.append(("DecTree Regressor ", DecisionTreeRegressor(random_state=707)))

#model = RandomForestRegressor(random_state = 707)
#model = DecisionTreeRegressor(random_state=707)
#model = SVR(C=1.0, epsilon=0.2)
lModels.append(("GradientBoostingRegressor  ",GradientBoostingRegressor(random_state=707)))

lModels.append(("AdaBoostRegressor ", AdaBoostRegressor(random_state=707, n_estimators=100)))
for vModel in lModels:
    print(vModel)
print("Done ...")


################################
# Regression - Cross Validation
###############################

# blank list to store results
print("\n*** Cross Validation Init ***")
xvModNames = []
xvRmseScrs = []
xvSDScores = []
print("Done ...")

# cross validation
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
#import sklearn
#print(sorted(sklearn.metrics.SCORERS.keys()))
print("\n*** Cross Validation ***")
# iterate through the lModels
for vModelName, oModelObj in lModels:
    # select xv folds
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=707)
    # actual corss validation
    cvRmse = cross_val_score(oModelObj, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
    # prints result of cross val ... scores count = lfold splits
    print(vModelName,":  ",cvRmse)
    # update lists for future use
    xvModNames.append(vModelName)
    xvRmseScrs.append(cvRmse.mean())
    xvSDScores.append(cvRmse.std())
    
# cross val summary
print("\n*** Cross Validation Summary ***")
# header
msg = "%16s: %10s %8s" % ("Model", "xvRM      ", "xvStdDev")
print(msg)
# for each model
for i in range(0,len(lModels)):
    # print rmse mean & std
    msg = "%10s: %5.7f %5.7f" % (xvModNames[i], xvRmseScrs[i], xvSDScores[i])
    print(msg)

# find model with best xv accuracy & print details
print("\n*** Best XV RMSE Model ***")
xvIndex = xvRmseScrs.index(max(xvRmseScrs))
print("Index      : ",xvIndex)
print("Model Name : ",xvModNames[xvIndex])
print("XVRMSE     : ",xvRmseScrs[xvIndex])
print("XVStdDev   : ",xvSDScores[xvIndex])
print("Model      : ",lModels[xvIndex])


###############################
# Split Train & Test
###############################

# split into data & target
print("\n*** Prepare Data ***")
dfTrain = df.sample(frac=0.8, random_state=707)
dfTest=df.drop(dfTrain.index)
print("Train Count:",len(dfTrain.index))
print("Test Count :",len(dfTest.index))

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
# evaluate test data with best model
###############################

# imports metrics
print("\n*** Import Metrics ***")
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print("Done ...")
 
# regression model
print("\n*** Regression Model ***")
# print model vals
print("Model Name: "+lModels[xvIndex][0])
# create model
model = lModels[xvIndex][1]
print(model)
# fit or train the model
model.fit(X_train, y_train) 
print("Done ...")

# predict train dataset 
print("\n*** Predict - Train Data ***")
p_train = model.predict(X_train)
dfTrain['predict'] = p_train
print("Done ...")

# predict test dataset
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
r2 = r2_score(y_train, p_train)
print(r2)

# adj r-square  
print('\n*** Adj R-Square ***')
adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))
print(adj_r2)

# mae 
print("\n*** Mean Absolute Error ***")
mae = mean_absolute_error(y_test, p_test)
print(mae)

# mse 
print("\n*** Mean Squared Error ***")
mse = mean_squared_error(y_test, p_test)
print(mse)
   
# rmse 
# RMSE measures the error.  How good is an error depends on the amplitude of your data. 
# RMSE should be less 10% for mean(depVars)
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# scatter index
# scatter index less than 1; the predictions are decent
print('\n*** Scatter Index ***')
si = rmse/y_test.mean()
print(si)


##############################################################
# predict from new data 
##############################################################

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
mname = lModels[xvIndex][0]
model = lModels[xvIndex][1]
model.fit(X,y)
print(mname)
print(model)

# read dataset
colNames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dfp = pd.read_csv('C:/Users/Manoj Yadav/Desktop/Part II/ML II/Regression/boston-housing-prd.dat', header=None, delimiter=r"\s+", names=colNames)

print("\n*** Structure ***")
print(dfp.info())

# drop cols - not required
print("\n*** Drop Cols ***")
print("N/A ... ")

# transformation
# change as required
print("\n*** Transformation ***")
print("None ... ")

# check nulls
print('\n*** Columns With Nulls ***')
print(dfp.isnull().sum()) 
print("Done ... ")

# split X & y
print("\n*** Split Predict Data ***")
print(allCols)
print(depVars)
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

# scatter index
# scatter index less than 1; the predictions are good
print('\n*** Scatter Index ***')
si = rmse/y_pred.mean()
print(si)
