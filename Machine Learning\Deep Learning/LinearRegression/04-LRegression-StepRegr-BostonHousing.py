
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

# check outlier index
print('\n*** Outlier Index ***')
import utils
print(utils.OutlierIndex(df))

# check outlier values
print('\n*** Outlier Values ***')
import utils
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


##############################################################
# Step Regression Function
# https://www.investopedia.com/terms/s/stepwise-regression.asp
##############################################################
import statsmodels.api as sm
def StepRegression(X, y,
                       initial_list=[], 
                       threshold_out = 0.05, 
                       verbose=True):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_out:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add ' + best_feature +' with p-value ' + str(best_pval))
        if not changed:
            break

    return included

################################
# Regression -  set X & y
###############################

# all cols except dep var 
print("\n*** Regression Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(depVars)
#allCols.sort()
print(allCols)
print("Done ...")

# train data
print("\n*** Split Regression Data In X & y ***")
X = df[allCols].values
y = df[depVars].values
print("Done ...")

# step regression for feature
print("\n*** Step Regression ***")
X = df[allCols]
y = df[depVars]
allCols = StepRegression(X,y)
#allCols.sort()
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

 
################################
# Regression - init models
###############################

# https://www.kaggle.com/jnikhilsai/cross-validation-with-linear-regression

# imports metrics
print("\n*** Import Metrics ***")
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print("Done ...")

# change as required
print("\n*** Regression Model ***")
# imports 
from sklearn.linear_model import LinearRegression 
# model names
vModelName = "Normal Linear Regression" 
# create model object
model = LinearRegression() 
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
mname = vModelName
model = LinearRegression() 
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
