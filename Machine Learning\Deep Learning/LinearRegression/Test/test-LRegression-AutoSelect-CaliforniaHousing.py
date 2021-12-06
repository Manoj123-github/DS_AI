
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
df = pd.read_csv('./data/california-housing.csv')


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
df = df.drop('ser', axis=1)
print("None ...")

# transformations
# change as required
print("\n*** Transformations ***")
# convert string / categoric to numeric
print("Unique ocean_proximity")
print(df['ocean_proximity'].unique())
from sklearn import preprocessing
leOpr = preprocessing.LabelEncoder()
df['ocean_proximity'] = leOpr.fit_transform(df['ocean_proximity'])
print(df['ocean_proximity'].unique())
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
# print('\n*** Normalize Data ***')
# df = utils.NormalizeData(df, depVars)
# print('Done ...')
# checked normalization does not inprove R-Square

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required
print('\n*** Handle Nulls ***')
# educated - to be dropped
df = df.drop('educated', axis=1)
# total rooms
vTotalRooms = min(df['total_rooms'].mean(),df['total_rooms'].median())
df['total_rooms'] = df['total_rooms'].fillna(vTotalRooms)
# total bed rooms
vTotalBedrooms = min(df['total_bedrooms'].mean(),df['total_bedrooms'].median())
df['total_bedrooms'] = df['total_bedrooms'].fillna(vTotalBedrooms)
# median_income
df['median_income'] = df['median_income'].interpolate(method ='linear', limit_direction ='forward') 
print('Done ...')

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.2f}'.format
dfc = df.corr()
print("Done ...")

# handle multi colinearity if required
# corr total_bedrooms & total_rooms : 0.927
# corr total_rooms & depVars : 0.045
# corr total_bedrooms & depVars : 0.069
df = df.drop('total_rooms', axis=1)
# corr household & population : 0.907
# corr household & depVara  : 0.055
# corr population & depVars : 0.099
df = df.drop('households', axis=1)
# corr total_rooms & population : 0.87
# drop col not required

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
colNames = ['ocean_proximity']
print("\n*** Distribution Plot ***")
for colName in colNames:
    plt.figure()
    sns.countplot(df[colName],label="Count")
    plt.title(colName)
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
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(dfTrain[allCols])
y = dfTrain[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# *** Regression Summary Extract ***
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:     median_house_value   R-squared:                       0.629
# Model:                            OLS   Adj. R-squared:                  0.629
# Method:                 Least Squares   F-statistic:                     3501.
# Date:                Sun, 23 May 2021   Prob (F-statistic):               0.00
# ======================================================================================
#                          coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# const              -3.809e+06   7.12e+04    -53.499      0.000   -3.95e+06   -3.67e+06
# random_income        -14.6601    188.763     -0.078      0.938    -384.657     355.337
# ocean_proximity     -448.4178    412.501     -1.087      0.277   -1256.964     360.129

# remove columns with p-value > 0.05
# change as required
print("\n*** Drop Cols ***")
allCols.remove('random_income')
allCols.remove('ocean_proximity')
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
print("Done ...")

# list model name list
print("\n*** Init Models Lists ***")
lModels.append(("LinearRegression", LinearRegression()))
lModels.append(("RidgeRegression ", Ridge(alpha = 10)))
lModels.append(("LassoRegression ", Lasso(alpha = 1)))
lModels.append(("ElasticNet      ", ElasticNet(alpha = 1)))
print("Done ...")

# imports
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# iterate through the models list
for vModelName, oModelObject in lModels:
    # create model object
    model = oModelObject
    # print model vals
    print("\n*** "+vModelName)
    # fit or train the model
    model.fit(X_train, y_train) 
    # predict train set 
    p_train = model.predict(X_train)
    dfTrain[vModelName] = p_train
    # predict test set 
    p_test = model.predict(X_test)
    dfTest[vModelName] = p_test
    # r-square  
    r2 = r2_score(y_train, p_train)
    print("R-Square:",r2)
    # adj r-square  
    adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
              (X_train.shape[0] - X_train.shape[1] - 1)))
    lModelAdjR2.append(adj_r2)
    print("Adj R-Square:",adj_r2)
    # mae 
    mae = mean_absolute_error(y_test, p_test)
    print("MAE:",mae)
    # mse 
    mse = mean_squared_error(y_test, p_test)
    print("MSE:",mse)
    # rmse 
    rmse = np.sqrt(mse)
    lModelRmses.append(rmse)
    print("RMSE:",rmse)
    # scatter index
    si = rmse/y_test.mean()
    lModelScInd.append(si)
    print("SI:",si)

# print key metrics for each model
print("\n*** Model Summary ***")
msg = "%10s %16s %10s %10s" % ("Model Type", "AdjR2", "RMSE", "SI")
print(msg)
for i in range(0,len(lModels)):
    # print accuracy mean & std
    msg = "%16s %10.3f %10.3f %10.3f" % (lModels[i][0], lModelAdjR2[i], lModelRmses[i], lModelScInd[i])
    print(msg)

# *** Model Summary ***
# Model Type            AdjR2       RMSE         SI
# LinearRegression      0.629  70106.876      0.334
# RidgeRegression       0.629  70107.603      0.334
# LassoRegression       0.629  70106.904      0.334
# ElasticNet            0.590  74072.992      0.353

# find model with best adj-r2 & print details
print("\n*** Best Model ***")
vBMIndex = lModelAdjR2.index(max(lModelAdjR2))
print("Index       : ",vBMIndex)
print("Model Name  : ",lModels[vBMIndex][0])
print("Adj-R-Sq    : ",lModelAdjR2[vBMIndex])
print("RMSE        : ",lModelRmses[vBMIndex])
print("ScatterIndex: ",lModelScInd[vBMIndex])

# *** Best Model ***
# Index       :  0
# Model Name  :  LinearRegression
# Adj-R-Sq    :  0.629077466859242
# RMSE        :  70106.87618092558
# ScatterIndex:  0.3341166866034387

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
mname = lModels[vBMIndex][0]
model = lModels[vBMIndex][1]
model.fit(X,y)
print(mname)
print(model)

# read dataset
dfp = pd.read_csv('./data/california-housing-prd.csv')
dfp = dfp.drop('ocean_proximity', axis=1)

print("\n*** Structure ***")
print(dfp.info())

# check nulls
print('\n*** Columns With Nulls ***')
print(dfp.isnull().sum()) 

# handle normalization if required
# print('\n*** Normalize Data ***')
# dfp = utils.NormalizeData(dfp, depVars)
# print('Done ...')

# split X & y
print("\n*** Split Predict Data ***")
X_pred = dfp[allCols].values
y_pred = dfp[depVars].values
print(X_pred)
print(y_pred)

# predict
print("\n*** Predict Data ***")
p_pred = model.predict(X_pred)
# read dataset again because we dont want naormalized data
dfp = pd.read_csv('./data/california-housing-prd.csv')
# upate predict
dfp['predict'] = p_pred
print("Done ... ")

# no y_pred values given
# so show predicted values
print("\n*** Print Predict Data ***")
for idx in dfp.index:
     print(dfp['ser'][idx], dfp['predict'][idx])
print("Done ... ")

# *** Print Predict Data ***
# 10001 180750.61138988286
# 10002 160311.04212807957
# 10003 170677.981204662
# 10004 195616.14519375283
# 10005 247816.48706375808
# 10006 260904.82409576885
# 10007 126130.36028680997
# 10008 86717.82387465192
# 10009 93166.10217614891
# 10010 162515.25497887377
# 10011 247646.7171457545
# 10012 54503.609939148184
# 10013 80572.1793876104
# 10014 72885.32628158573
# 10015 63068.753849619534
# Done ... 


