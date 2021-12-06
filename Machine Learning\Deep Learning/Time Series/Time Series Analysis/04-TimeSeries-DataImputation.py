

# https://www.machinelearningplus.com/time-series/time-series-analysis-python/

# imports
# warnings
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
plt.rcParams['figure.figsize'] = (8, 8)
# seaborn
import seaborn as sns
# mse
from sklearn.metrics import mean_squared_error

################################################################
# Missing Values 
################################################################
"""
Sometimes, your time series will have missing dates/times. That means, the data 
was not captured or was not available for those periods. It could so happen the 
measurement was zero on those days, in which case, case you may fill up those 
periods with zero.

Secondly, when it comes to time series, you should typically NOT replace missing 
values with the mean of the series, especially if the series is not stationary. 
What you could do instead for a quick and dirty workaround is to forward-fill or
backward-fill or interpolate (forward & backward)

However, depending on the nature of the series, you want to try out multiple 
approaches before concluding. 

Some effective alternatives to imputation are:
-- Forward Fill
-- Backward Fill
-- Linear Interpolation
-- Cubic Interpolation
-- Quadratic interpolation
-- Mean of nearest neighbors
-- Mean of seasonal couterparts

To measure the imputation performance, we manually introduce missing values to 
the time series, impute it with above approaches and then measure the mean 
squared error of the imputed against the actual values.
"""

################################################################
# read data
################################################################
from datetime import datetime
custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
# read files
df = pd.read_csv('C:/Users/Manoj Yadav/Desktop/ML/Time Series/Time Series Analysis/synth-data.csv', parse_dates=['date'], index_col='date', 
                date_parser=custom_date_parser)


################################################################
# EDA
################################################################

# info
print("\n*** Structure ***")
print(df.info())

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# head
print("\n*** Head ***")
print(df.head())

# tail
print("\n*** Tail ***")
print(df.tail())

# actual v/s missing
# notice the legends are opposite
print("*** Actual v/s Missing Data ***")
plt.figure(figsize=(10,4))
sns.lineplot(x=df.index, y='value', data=df, color='red')
sns.lineplot(x=df.index, y='nulls', data=df, color='green')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Actual v/s Missing Data")
plt.legend(["Missing Data","Actual Data"])
plt.show()

################################################################
# Imputations
################################################################

# forward fill 
print("*** Forward Fill Data ***")
df['ffill'] = df['nulls'].ffill()
errMSE = np.round(mean_squared_error(df['value'], df['ffill']), 2)
plt.figure(figsize=(10,4))
sns.lineplot(x=df.index, y='value', data=df, color='red')
sns.lineplot(x=df.index, y='ffill', data=df, color='blue')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Forward Fill Data. MSE: "+str(errMSE))
plt.legend(["Missing Data","Forward Fill"])
plt.show()


# backward fill 
print("*** Backward Fill Data ***")
df['bfill'] = df['nulls'].bfill()
errMSE = np.round(mean_squared_error(df['value'], df['bfill']), 2)
plt.figure(figsize=(10,4))
sns.lineplot(x=df.index, y='value', data=df, color='red')
sns.lineplot(x=df.index, y='bfill', data=df, color='blue')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Backward Fill Data. MSE: "+str(errMSE))
plt.legend(["Missing Data","Backward Fill"])
plt.show()


# linear interpolation 
print("*** Linear Interpolation ***")
df['lintr'] = df['nulls'].interpolate(method ='linear')
errMSE = np.round(mean_squared_error(df['value'], df['lintr']), 2)
plt.figure(figsize=(10,4))
sns.lineplot(x=df.index, y='value', data=df, color='red')
sns.lineplot(x=df.index, y='lintr', data=df, color='blue')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Linear Interpolation Data. MSE: "+str(errMSE))
plt.legend(["Missing Data","Linear Interpolation"])
plt.show()


# cubic interpolation 
print("*** Cubic Interpolation ***")
df['cintr'] = df['nulls'].interpolate(method ='cubic')
errMSE = np.round(mean_squared_error(df['value'], df['cintr']), 2)
plt.figure(figsize=(10,4))
sns.lineplot(x=df.index, y='value', data=df, color='red')
sns.lineplot(x=df.index, y='cintr', data=df, color='blue')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Cubic Interpolation Data. MSE: "+str(errMSE))
plt.legend(["Missing Data","Cubic Interpolation"])
plt.show()


# quadratic interpolation 
print("*** Quadratic Interpolation ***")
df['qintr'] = df['nulls'].interpolate(method ='quadratic')
errMSE = np.round(mean_squared_error(df['value'], df['qintr']), 2)
plt.figure(figsize=(10,4))
sns.lineplot(x=df.index, y='value', data=df, color='red')
sns.lineplot(x=df.index, y='qintr', data=df, color='blue')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Quadratic Interpolation Data. MSE: "+str(errMSE))
plt.legend(["Missing Data","Quadratic Interpolation"])
plt.show()


# get KNN_Mean function
def getKNN_Mean(ts, n):
    """
    Compute the mean of K nearest rows up & down
    ts: 1D array-like of the time series
    n: number of nearest rows
    """
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            n_by_2 = np.ceil(n/2)
            lower = np.max([0, int(i-n_by_2)])
            upper = np.min([len(ts)+1, int(i+n_by_2)])
            ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
            out[i] = np.nanmean(ts_near)
    return out

# Mean of 'k' Nearest Neighbors
print("*** KNN Mean ***")
df['kmean'] = getKNN_Mean(df['nulls'], 6)
errMSE = np.round(mean_squared_error(df['value'], df['kmean']), 2)
plt.figure(figsize=(10,4))
sns.lineplot(x=df.index, y='value', data=df, color='red')
sns.lineplot(x=df.index, y='kmean', data=df, color='blue')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("K Nearest Neighbour Data. MSE: "+str(errMSE))
plt.legend(["Missing Data","K Nearest Neighbour"])
plt.show()


# getSeasonalMean function
def getSeasonalMean(ts, n):
    """
    Compute the mean of corresponding seasonal periods
    ts: 1D array-like of the time series
    n: Seasonal window length of the time series
    """
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            ts_seas = ts[i-1::-n]  # previous seasons only
            if np.isnan(np.nanmean(ts_seas)):
                ts_seas = np.concatenate([ts[i-1::-n], ts[i::n]])  # previous and forward
            out[i] = np.nanmean(ts_seas)
    return out

# seasonal mean
print("*** Seasonal Mean ***")
df['smean'] = getSeasonalMean(df['nulls'], n=12)
errMSE = np.round(mean_squared_error(df['value'], df['smean']), 2)
plt.figure(figsize=(10,4))
sns.lineplot(x=df.index, y='value', data=df, color='red')
sns.lineplot(x=df.index, y='smean', data=df, color='blue')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Seasonal Mean. MSE: "+str(errMSE))
plt.legend(["Missing Data","Seasonal Mean"])
plt.show()
