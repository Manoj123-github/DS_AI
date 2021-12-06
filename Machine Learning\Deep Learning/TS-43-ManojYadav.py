# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 15:24:29 2021

@author: Manoj Yadav
"""

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
df = pd.read_csv("C:/Users/Manoj Yadav/Desktop/ML/adidas-sales.csv")
vTitle = "Adidas-sales"
vColDate = "Period"
vColData = "Revenue"
vDatFrmt = "%Y-%m"  

print("\n*** Input ***")
print("Title:",vTitle)
print("Quarter:",vColDate)
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

# set the 'Period' as index
#df = df.set_index(vColDate)
df.set_index(vColDate, inplace=True)
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

# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# set the frequency of the date time index as Monthly start as indicated by the data
df.index.freq = 'Q'
# define m as the time period
m = 4
# Set the value of Alpha 
alpha = 1/(2*m)

# triple exponentail smoothing - additive
df['TESA'] = ExponentialSmoothing(df['Revenue'],trend='add',seasonal='add',seasonal_periods=4).fit().fittedvalues
# triple exponentail smoothing - multiplicative
df['TESM'] = ExponentialSmoothing(df['Revenue'],trend='mul',seasonal='mul',seasonal_periods=4).fit().fittedvalues

# plot
print("\n*** Plot Time Series ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=df.index, y=vColData, data=df, color='k',label='Original')
sns.lineplot(x=df.index, y='TESA', data=df, color='r',label='TESA')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle)
plt.show()

# plot
print("\n*** Plot Time Series ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=df.index, y=vColData, data=df, color='k',label='Original')
sns.lineplot(x=df.index, y='TESM', data=df, color='r',label='TESM')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle)
plt.show()
 
##############################################################
# forecast - test using additive
##############################################################

# split data into dfTrain / dfTest sets 
print("\n*** Split In dfTrain & dfTest ***")
dfTrain = df.iloc[:len(df)-4] 
dfaTest = df.iloc[len(df)-4:] # set quartly for dfTesting 
print("Done ...")
  
# dfTrain the model
print("\n*** Model - Create & Train ***")
model = ExponentialSmoothing(dfTrain[vColData],trend='add',seasonal='add',
                             seasonal_periods=4).fit()
print("Done ...")
 
# predictions for one-year against the dfaTest set 
print("\n*** Model - Forecast ***")
dfaTest['Forecast'] = model.forecast(m)
print("Done ...")

# z-score for prediction interval
# CI	z-score            CI	z-score            CI	z-score 
# 50%	0.674              80%	1.282              90%	1.645
# 95%	1.96               98%	2.326              99%	2.576

# prediction interval
print("\n*** Model Forecast - PredInt ***")
# z-score
z = 1.645
# sum of square errors
sse = model.sse
# pred interval
dfaTest['LoPredInt'] = dfaTest['Forecast'] - z * np.sqrt(sse/len(dfTrain))
dfaTest['UpPredInt'] = dfaTest['Forecast'] + z * np.sqrt(sse/len(dfTrain))
print("Done ...")

# print
print("\n*** Model Forecast - Print ***")
print(dfaTest[[vColData,'LoPredInt','Forecast','UpPredInt']])

# plot
print("\n*** Plot Time Series - dfaTest ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=dfTrain.index, y=vColData, data=dfTrain, label='dfTrain', color='k')
sns.lineplot(x=dfaTest.index, y=vColData, data=dfaTest, label='dfaTest', color='r')
sns.lineplot(x=dfaTest.index, y="Forecast", data=dfaTest, label='dfaTest.Forecast', color='m')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle + " - Plot Time Series - dfaTest")
plt.show()

# calculate mean squared error 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error 
dfaMse = mean_squared_error(dfaTest[vColData], dfaTest['Forecast']) 
print(dfaMse)

# calculate Root mean squared error 
print("\n*** Root Mean Squared Error ***")
dfaRmse = np.sqrt(dfaMse)
print (dfaRmse)

# scatter index
# scatter index less than 1; the predictions are good
print('\n*** Scatter Index ***')
dfaSi = dfaRmse/dfaTest[vColData].mean()
print(dfaSi)


##############################################################
# forecast test date series - using multiplicative
##############################################################

# split data into dfTrain / dfmTest sets 
print("\n*** Split In dfTrain & dfmTest ***")
dfTrain = df.iloc[:len(df)-4] 
dfmTest = df.iloc[len(df)-4:] # set quartly for dfTesting 
print("Done ...")
  
# dfTrain the model
print("\n*** Model - Create & Train ***")
model = ExponentialSmoothing(dfTrain[vColData],trend='mul',seasonal='mul',
                             seasonal_periods=4).fit()
print("Done ...")
 
# predictions for one-year against the dfmTest set 
print("\n*** Model - Forecast ***")
dfmTest['Forecast'] = model.forecast(m)
print("Done ...")

# z-score for prediction interval
# CI	z-score            CI	z-score            CI	z-score 
# 50%	0.674              80%	1.282              90%	1.645
# 95%	1.96               98%	2.326              99%	2.576
# prediction interval
print("\n*** Model Forecast - PredInt ***")
# z-score
z = 1.645
# sum of square errors
sse = model.sse
# pred interval
dfmTest['LoPredInt'] = dfmTest['Forecast'] - z * np.sqrt(sse/len(dfTrain))
dfmTest['UpPredInt'] = dfmTest['Forecast'] + z * np.sqrt(sse/len(dfTrain))
print("Done ...")

# print
print("\n*** Model Forecast - Print ***")
print(dfmTest[[vColData,'LoPredInt','Forecast','UpPredInt']])

# plot
print("\n*** Plot Time Series - dfmTest ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=dfTrain.index, y=vColData, data=dfTrain, label='dfTrain', color='k')
sns.lineplot(x=dfmTest.index, y=vColData, data=dfmTest, label='dfmTest', color='r')
sns.lineplot(x=dfmTest.index, y="Forecast", data=dfmTest, label='dfmTest.Forecast', color='m')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle + " - Plot Time Series - dfmTest")
plt.show()

# calculate mean squared error 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error 
dfmMse = mean_squared_error(dfmTest[vColData], dfmTest['Forecast']) 
print(dfmMse)

# calculate Root mean squared error 
print("\n*** Root Mean Squared Error ***")
dfmRmse = np.sqrt(dfmMse)
print (dfmRmse)

# scatter index
# scatter index less than 1; the predictions are good
print('\n*** Scatter Index ***')
dfmSi = dfmRmse/dfmTest[vColData].mean()
print(dfmSi)

##############################################################
# forecast future date series - using besr of additive / multiplicative
##############################################################

# print dfa metrics
print("\n*** Metrics - Additive Model ***")
print("MSE  : ", dfaMse)
print("RMSE : ", dfaRmse)
print("SI   : ", dfaSi)

# print dfm metrics
print("\n*** Metrics - Multiplicative Model ***")
print("MSE  : ", dfmMse)
print("RMSE : ", dfmRmse)
print("SI   : ", dfmSi)

# split data into dfTrain / dfTest sets 
print("\n*** Split In dfTrain & dfmTest ***")
dfTrain = df.iloc[:len(df)-4] 
dfTest = df.iloc[len(df)-4:] # set quartly for dfTesting 
print("Done ...")
  
# compare and build final test model
# dfTrain the model
print("\n*** Model - Create & Train ***")
if dfaRmse <= dfmRmse:
    model = ExponentialSmoothing(dfTrain[vColData],trend='add',seasonal='add',
                                 seasonal_periods=4).fit()
else:
    model = ExponentialSmoothing(dfTrain[vColData],trend='mul',seasonal='mul',
                                 seasonal_periods=4).fit()
print("Done ...")
 
# predictions for one-year against the dfmTest set 
print("\n*** Model - Forecast ***")
dfTest['Forecast'] = model.forecast(m)
print("Done ...")

# future date list
print("\n*** Dates For New Data ***")
lsPred =  tsutils.create_future_date_list(dfTest.index[-1],n=4,p='Months')
for vPred in lsPred:
    print(vPred)

# convert list to dataframe
print("\n*** Dataframe For New Data ***")
dfPred = pd.DataFrame(lsPred,columns=[vColDate])
dfPred[vColData] = None
print(dfPred.head())

# set the 'Month' as index
print("\n*** Dataframe For New Data - Again ***")
dfPred.set_index(vColDate, inplace=True)
print(dfPred.head())

# predictions for one-year against the predict set
print("\n*** Model - New Data - Forecast ***")

lForecast = model.forecast(len(dfTest)+len(dfPred))
# why -4 - last 4 will be data from quartely months of predict
dfPred['Forecast'] = lForecast[-4:]
dfPred['LoPredInt'] = dfPred['Forecast'] * 0.95
dfPred['UpPredInt'] = dfPred['Forecast'] * 1.05
dfPred = dfPred[['LoPredInt','Forecast','UpPredInt']]
print("Done ...")

# prediction interval
print("\n*** Model - New Data - PredInt ***")
# z-score
z = 1.645
# sum of square errors
sse = model.sse
# pred interval
dfPred['LoPredInt'] = dfPred['Forecast'] - z * np.sqrt(sse/len(dfTrain))
dfPred['UpPredInt'] = dfPred['Forecast'] + z * np.sqrt(sse/len(dfTrain))
print("Done ...")

# print
print("\n*** Model - New Data - Forecast Print ***")
print(dfPred[['LoPredInt','Forecast','UpPredInt']])

# plot
print("\n*** Plot dfPred Time Series ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=dfTrain.index, y=vColData, data=dfTrain, label='dfTrain', color='k')
sns.lineplot(x=dfTest.index, y=vColData, data=dfTest, label='dfTest', color='b')
sns.lineplot(x=dfTest.index, y="Forecast", data=dfTest, label='dfTest.Forecast', color='m')
sns.lineplot(x=dfPred.index, y="Forecast", data=dfPred, ci=0.95, label='dfPred.Forecast', color='r')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle)
plt.show()
