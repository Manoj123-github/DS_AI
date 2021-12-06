

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
# math
import math 
# utils
import utils

##############################################################
# Read Data 
##############################################################

# read dataset
df = pd.read_csv('./data/slr-manual-data.csv')
df = pd.read_csv('./data/slr-salary-data.csv')


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
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
#df = df.drop('Month', axis=1)
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

# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.2f}'.format
print(df.corr())

# handle multi colinearity if required


##############################################################
# Visual Data Analytics
##############################################################

# scatter plot - hits & rev - spl case
print('\n*** Scatterplot ***')
plt.figure()
sns.regplot(x=df.iloc[:,0], y=df.iloc[:,0], data=df, color= 'b', scatter_kws={"s": 10})
plt.title('X v/s Y')
plt.ylabel('X')
plt.xlabel('Y')
# good practice
plt.show()


##########################################################
# Ideally we should use object oriented programming but we dont know how to do that
# so we will create a function where we do all computations and display the same
# commented print statements may be remove ... included as it helps debugging
##############################################################
def FunctionLinearRegression(pX, pY, pV=[]):
    """
    desc:
        returns important metrics of simple linear regression algo
        OR
        returns predicted data is predict X is passed
    usage: 
        FunctionLinearRegression(X-Col, Y-Col, [P-Col]) 
    params:
        X-Col: X part of the Linear Regression  
        Y-Col: Y part of the Linear Regression  
        P-Col: data for predictions
    returns:
        if P-Col passed as param
            LR-Data: important metrics of simple linear regression algo
        if P-Col NOT passed as param
            P-Val: predicted values
    """
    #px = df['Spend (x)']
    #py = df['Sales (y)']
    #print(pX)
    #print(pX)
    # format
    pd.options.display.float_format = '{:,.4f}'.format
    # data series
    dsRetValue = pd.Series() 
    # create temp data frame
    dt = { 'X': pX, 'Y': pY }
    dfLR = pd.DataFrame(dt)
    #print(dfLR)
    # compute r-square #######################################################
    # create cols required for r2
    dfLR['XY'] = dfLR['X'] * dfLR['Y']
    dfLR['X2'] = dfLR['X'] ** 2  
    dfLR['Y2'] = dfLR['Y'] ** 2  
    #print(dfLR)
    # compute R in small formulas rather than 1 big formula
    vR1 = (dfLR['X'].size * dfLR['XY'].sum())  
    vR2 = (dfLR['X'].sum() * dfLR['Y'].sum())  
    vR12 = (vR1-vR2)
    #print(vR1) 
    #print(vR2) 
    #print(vR12) 
    vR3 = (dfLR['X'].size * dfLR['X2'].sum())-(dfLR['X'].sum()**2) 
    vR4 = (dfLR['X'].size * dfLR['Y2'].sum())-(dfLR['Y'].sum()**2) 
    vR34 = math.sqrt(float(vR3)*float(vR4))
    #print(vR3) 
    #print(vR4) 
    #print(vR34) 
    vR   = vR12 / vR34
    dsRetValue['r'] = vR
    vR2 = vR ** 2
    dsRetValue['r2'] = vR2
    #print("r             : ", vR)
    #print("r-square      : ", vR2)
    # compute slope & intercept  #############################################
    vAvX = dfLR['X'].mean()
    vAvY = dfLR['Y'].mean()
    #dfLR['FXX'] = 0
    #dfLR['FXX'] = dfLR['FXX'].astype(np.float64)
    #dfLR['FXY'] = 0
    #dfLR['FXY'] = dfLR['FXY'].astype(np.float64)
    dfLR['FXX'] = ( vAvX - dfLR['X'] ) ** 2
    dfLR['FXY'] = ( vAvX - dfLR['X'] ) * ( vAvY - dfLR['Y'])
    #print( np.int64(dfLR['FXX'].sum() ) )
    #print( np.int64(dfLR['FXY'].sum() ) )
    vSlope = np.float64(np.int64(dfLR['FXY'].sum() ) / np.int64(dfLR['FXX'].sum() ) )
    dsRetValue['slope'] = vSlope
    #print("slope         : ", vSlope)
    #print( dfLR['X'].mean() )
    #print( dfLR['Y'].mean() )
    vIcept = dfLR['Y'].mean() - ( vSlope * dfLR['X'].mean()  ) 
    dsRetValue['intercept'] = vIcept
    #print("intercept     : ", vIcept)
    # compute rmse & si  #####################################################
    dfLR['P'] = ( vSlope * dfLR['X'] ) + vIcept
    dfLR['R'] = ( dfLR['Y'] - dfLR['P'] ) ** 2
    vRmse = math.sqrt(dfLR['R'].mean())
    dsRetValue['rmse'] = vRmse
    #print("rmse          : ",vRmse)
    vSI = vRmse / dfLR['Y'].mean()
    dsRetValue['SI'] = vSI
    #print("scatter index : ",vSI)
    if (len(pV) == 0):
        return dsRetValue
    else:
        vP = (vSlope * pV) + vIcept 
        return vP

# call the function without data to be predicted
# call function using column names
#lr = FunctionLinearRegression(df['Spend (x)'], df['Sales (y)'])
# call function using column index
lr = FunctionLinearRegression(df.iloc[:,0], df.iloc[:,1])
print(lr)

# predict for below 
vP = np.array([8000,9000])
# for salary data
#vP = np.array([2.5,3.5,4.5,5.5])

# call the function with data to be predicted
#pP = FunctionLinearRegression(df['Spend (x)'], df['Sales (y)'], vP)
pP = FunctionLinearRegression(df.iloc[:,0], df.iloc[:,1],vP)
print(vP)
print(pP)

        
    