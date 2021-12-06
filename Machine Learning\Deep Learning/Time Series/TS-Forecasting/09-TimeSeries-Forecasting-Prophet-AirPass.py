

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

# for Prophet, the time series data should be in two cols
# ds - date series (should be column not index)
# y  - data (MUST be float)
# column names MUST be as above

# covert to datetime
print("\n*** Convert To DateTime ***")
df[vColDate] = pd.to_datetime(df[vColDate],format=vDatFrmt)
print(df.info())

# set the 'Month' as index for df-temp
print("\n*** Set Index ***")
dft = df.set_index(vColDate)
#df.set_index(vColDate, inplace=True)
print(df.head())

# Augmented Dickey-Fuller Test
print('\n*** ADF Test - df ***')
print(tsutils.adfTest(dft))

# prophet specific pre-process
print('\n*** Prophet Specific Pre-process ***')
# convert to float
df[vColData] = df[vColData].astype(np.float64)
# rename cols
df = df.rename(columns={vColDate:'ds', vColData: 'y'})
print("Done ...")

# info
print("\n*** Structure Again ***")
print(df.info())

# plot
print("\n*** Plot Time Series ***")
plt.figure(figsize=(9,4))
sns.lineplot(x='ds', y='y', data=df, label="Data", color='b')
plt.xticks(rotation=60)
plt.title(vTitle)
plt.legend()
plt.show()

# time series model - create & train
print("\n*** Prophet Model ***")
# set the confidence interval to 95% (the Prophet default is 80%)
from fbprophet import Prophet
model = Prophet(interval_width=0.95)
# fit tsm
model.fit(df)
print("Done ...")

# forcast test data ... here our test data is entire df
print("\n*** Model Forecast ***")
dfTest = model.predict(df)
print("Done ...")

# prophet forecast plot
# actual - black dots
# forecast = blue line
# pred-int = blue band
print("\n*** Forecast Plot ***")
plt.figure(figsize=(9,4))
model.plot(dfTest)
plt.title(vTitle + " - Prophet Plot")
plt.legend()
plt.show()

# forecast test dataframe
print("\n*** Forecast Dataframe ***")
# merge on col
dfTest = pd.merge(dfTest, df, on='ds', how='left')
# forecast head & tail
print(dfTest[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].head())
print(dfTest[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# sns plot
print("\n*** Seaborn Forecast Plot ***")
plt.figure(figsize=(9,4))
sns.lineplot(x='ds', y='y', data=dfTest, label='Actual', color='k')
sns.lineplot(x='ds', y='yhat', data=dfTest, label='Forecast', color='m')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle + " - Seaborn Plot")
plt.show()

# calculate mean squared error 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error 
mse = mean_squared_error(dfTest['y'], dfTest['yhat']) 
print(mse)

# calculate Root mean squared error 
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print (rmse)

# scatter index
# scatter index less than 1; the predictions are good
print('\n*** Scatter Index ***')
si = rmse/dfTest['y'].mean()
print(si)

# forecast - future dates
# predict data ... includes entire df + future dates
# https://rdrr.io/cran/prophet/man/make_future_dataframe.html
print("\n*** Future Forecast Prepare Data ***")
fPeriods = 12           # change as required
fFreqs = 'MS'           # change as required
dfPred = model.make_future_dataframe(periods=fPeriods, freq=fFreqs)
print(dfPred.tail(fPeriods))
print("\nDone ... ")

# predict
print("\n*** Model Future Forecast ***")
dfPred = model.predict(dfPred)
print(dfPred.tail(fPeriods))

# forecaste dataframe
print("\n*** Model Future Forecast Dataframe ***")
# merge
dfPred = pd.merge(dfPred, df, on='ds', how='left')
# forecast
print(dfPred[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].tail(fPeriods+12))

# plot
print("\n*** Prophet Forecast Plot ***")
plt.rcParams['figure.figsize'] = (8, 5)
plt.figure(figsize=(8,5))
model.plot(dfPred, uncertainty=True)
plt.title(vTitle + " - Prophet Plot")
plt.show()

# plot
print("\n*** Seaborn Forecast Plot ***")
plt.figure(figsize=(9,4))
sns.lineplot(x='ds', y='y', data=dfTest, label='Actual', color='k')
sns.lineplot(x='ds', y='yhat', data=dfPred, label='Forecast', color='m')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle + " - Seaborn Plot")
plt.show()


