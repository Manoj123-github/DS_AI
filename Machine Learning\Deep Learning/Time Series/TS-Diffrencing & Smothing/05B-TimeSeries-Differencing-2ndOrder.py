

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
# utils for ts
import tsutils

# one of the most common methods of dealing with both trend and seasonality is 
# differencing. In this technique, we take the difference of the observation at 
# a particular instant with that at the previous instant. This mostly works well 
# in improving stationarity.

###############################################################
# detrend by differencing
###############################################################

# read data
df = pd.read_csv("./data/air-passengers.csv")
vTitle = "Air Passengers"
vColDate = "Month"
vColData = "Passengers"
vDatFrmt = "%Y-%m"  

print("\n*** Input ***")
print("Title:",vTitle)
print("Month:",vColDate)
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
if df[vColDate].isnull().sum() > 0:
    df = df.dropna(subset=['Month'])
# if data column is null interpolate
if df[vColData].isnull().sum() > 0:
    df[vColData] = df[vColData].interpolate(method ='linear')
print(df.isnull().sum()) 

# covert to datetime
df[vColDate] = pd.to_datetime(df[vColDate],format=vDatFrmt)
print("\n*** Structure Again ***")
print(df.info())

# set the 'Month' as index
#df = df.set_index(vColDate)
df.set_index(vColDate, inplace=True)
print("\n*** Head Again ***")
df.head()

# Augmented Dickey-Fuller Test
print('\n*** ADF Test - df ***')
print(tsutils.adfTest(df))

# plot
print("\n*** Plot Time Series - df ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=df.index, y=vColData, label=vColData, data=df, color='b')
plt.xticks(rotation=60)
plt.title(vTitle)
plt.legend()
plt.show()

# differencing - first order
print("\n*** Differencing First Order ***")
d1 = df - df.shift()
d1.dropna(inplace=True)
print("Done ...")

# Augmented Dickey-Fuller Test
print('\n*** ADF Test - d1 ***')
print(tsutils.adfTest(d1))

# plot
print("\n*** Plot Time Series - d1 ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=df.index, y=vColData, data=df, color='b')
sns.lineplot(x=d1.index, y=vColData, data=d1, color='c')
plt.xticks(rotation=60)
plt.title('Differencing First Order')
plt.show()

# detrend by differencing - second order
d2 = d1 - d1.shift()
d2.dropna(inplace=True)

# Augmented Dickey-Fuller Test
print('\n*** ADF Test - d2 ***')
print(tsutils.adfTest(d2))

# plot
print("\n*** Plot Time Series - d1 ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=df.index, y=vColData, data=df, color='b')
sns.lineplot(x=d2.index, y=vColData, data=d2, color='c')
plt.xticks(rotation=60)
plt.title('Differencing Second Order')
plt.show()

