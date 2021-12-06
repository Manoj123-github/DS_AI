

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
#%matplotlib
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8, 8)
# seaborn
import seaborn as sns
# utils for ts
import tsutils
   
# read data
df = pd.read_csv("./data/air-passengers.csv")
vTitle = "Air Passengers"
pColDate = "Month"
vColData = "Passengers"
vDatFrmt = "%Y-%m"  

print("\n*** Input ***")
print("Title:",vTitle)
print("Month:",pColDate)
print("Data :",vColData)
print("Frmt :",vDatFrmt)

# info
print("\n*** Structure ***")
print(df.info())

# head
print("\n*** Head ***")
print(df.head())

# tail
print("\n*** Tail ***")
print(df.tail())

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required
print("\n*** Handle Nulls ***")
# if date column is null drop NaN row
if df[pColDate].isnull().sum() > 0:
    df = df.dropna(subset=['Month'])
# if data column is null interpolate
if df[vColData].isnull().sum() > 0:
    df[vColData] = df[vColData].interpolate(method ='linear')
print(df.isnull().sum()) 

# covert to datetime
df[pColDate] = pd.to_datetime(df[pColDate],format=vDatFrmt)
print("\n*** Structure Again ***")
print(df.info())

# set the 'Month' as index
#df = df.set_index(pColDate)
df.set_index(pColDate, inplace=True)
print("\n*** Head Again ***")
df.head()

# plot
print("\n*** Plot Time Series ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=df.index, y=vColData, data=df, color='b')
plt.xticks(rotation=60)
plt.title(vTitle)
plt.show()

##############################################################
# smoothing
##############################################################

# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# set the frequency of the date time index as Monthly start as indicated by the data
df.index.freq = 'MS'
# define m as the time period
m = 12
# Set the value of Alpha 
# ALPHA is the smoothing parameter that defines the weighting and should be 
# greater than 0 and less than 1. ALPHA equal 0 sets the current smoothed point 
# to the previous smoothed value and ALPHA equal 1 sets the current smoothed 
# point to the current point (i.e., the smoothed series is the original series). 
# The closer ALPHA is to 1, the less the prior data points enter into the smooth. 
# In practice, wet set Alpha to 1/(2*m) 
alpha = 1/(2*m)

# single exponential smoothing
df['SES'] = SimpleExpSmoothing(df[vColData]).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues

# plot
print("\n*** Plot Time Series ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=df.index, y=vColData, data=df, color='k',label='Original')
sns.lineplot(x=df.index, y='SES', data=df, color='r',label='HWSES')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle)
plt.show()

# double exponential smoothing - additive
df['DESA'] = ExponentialSmoothing(df[vColData],trend='add').fit().fittedvalues
# double exponential smoothing - multiplicative
df['DESM'] = ExponentialSmoothing(df[vColData],trend='mul').fit().fittedvalues

# plot
print("\n*** Plot Time Series ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=df.index, y=vColData, data=df, color='k',label='Original')
sns.lineplot(x=df.index, y='DESA', data=df, color='r',label='DESA')
sns.lineplot(x=df.index, y='DESM', data=df, color='b',label='DESM')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle)
plt.show()

# triple exponentail smoothing - additive
df['TESA'] = ExponentialSmoothing(df[vColData],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
# triple exponentail smoothing - multiplicative
df['TESM'] = ExponentialSmoothing(df[vColData],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues

# plot
print("\n*** Plot Time Series ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=df.index, y=vColData, data=df, color='k',label='Original')
sns.lineplot(x=df.index, y='TESA', data=df, color='r',label='TESA')
sns.lineplot(x=df.index, y='TESM', data=df, color='b',label='TESM')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle)
plt.show()

