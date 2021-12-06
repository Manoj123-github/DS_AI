
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

# covert to datetime
df[vColDate] = pd.to_datetime(df[vColDate],format=vDatFrmt)
print("\n*** Structure Again ***")
print(df.info())

# set the 'Month' as index
#df = df.set_index(vColDate)
df.set_index(vColDate, inplace=True)
print("\n*** Head Again ***")
df.head()

# Augmented Dickey-Fuller dfTest
print('\n*** ADF dfTest ***')
print(tsutils.adfTest(df))

# plot
print("\n*** Plot Time Series ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=df.index, y=vColData, label=vColData, data=df, color='b')
plt.xticks(rotation=60)
plt.title(vTitle)
plt.legend()
plt.show()

##############################################################
# parameter analysis for the ARIMA tsm
##############################################################

"""
An ARIMA model is a class of statistical models for analyzing and forecasting 
time series data.

It explicitly caters to a suite of standard structures in time series data, and 
as such provides a simple yet powerful method for making skillful time series forecasts.

ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average. 
It is a generalization of the simpler AutoRegressive Moving Average and adds the 
notion of integration.

This acronym is descriptive, capturing the key aspects of the model itself. 
Briefly, they are:
AR: Autoregression. A model that uses the dependent relationship between an 
    observation and some number of lagged observations.
I : Integrated. The use of differencing of raw observations (e.g. subtracting 
    an observation from an observation at the previous time step) in order to 
    make the time series stationary.
MA: Moving Average. A model that uses the dependency between an observation and 
    a residual error from a moving average model applied to lagged observations.
Each of these components are explicitly specified in the model as a parameter. 
A standard notation is used of ARIMA(p,d,q) where the parameters are substituted 
with integer values to quickly indicate the specific ARIMA model being used.

The parameters of the ARIMA model are defined as follows:
p: The number of lag observations included in the model, also called the lag order.
d: The number of times that the raw observations are differenced, also called the degree of differencing.
q: The size of the moving average window, also called the order of moving average.

A linear regression model is constructed including the specified number and type 
of terms, and the data is prepared by a degree of differencing in order to make 
it stationary, i.e. to remove trend and seasonal structures that negatively affect 
the regression model.
"""
# to install the library 
#!pip install pmdarima 
  
"""
SARMIA - seasonal arima 
requires p, d, q  for normal (level) so provide start & max p & q 
requires P, D, Q  for season so provide start & max P & Q
"""
# fit auto_arima function to dataset 
print("\n*** Model Order Parameters ***")
from pmdarima import auto_arima 
oModelOrder = auto_arima(df[vColData], 
                          start_p = 0, max_p = 5, 
                          start_q = 0, max_q = 5, 
                          d = None, D = 1, m = 12, 
                          start_P = 0, max_P = 5, 
                          start_Q = 0, max_Q = 5, 
                          seasonal = True, 
                          random_state = 707,
                          trace = True, 
                          error_action ='ignore',    # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise 

# Akaike's Information Criterion (AIC), is useful in selecting or determining 
# the order (p,d,q) of an ARIMA model. The combination with lowest AIC is best
print("\n*** Model Order Best ***")
#print(oModelOrder)
vBestOrder = str(oModelOrder)
#print(vBestOrder)
lBestOrder = vBestOrder.split(')')
#print(lBestOrder)
lBestOrder[0] = lBestOrder[0].replace(' ARIMA(','')
lBestOrder[1] = lBestOrder[1].replace('(','')
lBestOrder[2] = lBestOrder[2].replace('[','')
lBestOrder[2] = lBestOrder[2].replace(']','').strip()
#print(lBestOrder)
# normal order
nOrder = tuple(map(int, lBestOrder[0].split(',')))
# seasonal order
sOrder = tuple(map(int, lBestOrder[1].split(',')))
sOrder = list(sOrder)
sOrder.append(int(lBestOrder[2]))
sOrder = tuple(sOrder)
print('Normal Order : ', nOrder)
print('Season Order : ', sOrder)

# to print the summary 
print("\n*** Best Order Summary ***")
oModelOrder.summary() 

# split data into dfTrain / dfTest sets 
print("\n*** Split In dfTrain & dfTest ***")
dfTrain = df.iloc[:len(df)-12] 
dfTest = df.iloc[len(df)-12:] # set one year(12 months) for dfTesting 
print("Done ...")
  
# fit a SARIMAX(0, 1, 1)x(2, 1, 0, 12) on the dfTraining set 
# create tsm with best tsmparameters
print("\n*** Time Series Model - Create ***")
warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX 
model = SARIMAX(dfTrain[vColData],  
                order = nOrder,  
                seasonal_order = sOrder) 
print("Done ...")

# dfTrain model
print("\n*** Time Series Model - dfTrain ***")
vResult = model.fit() 
print(vResult.summary())

# prepare start end to match with dfTest
# required in period serial number 
print("\n*** Time Series Model - Period Serial Nos ***")
start = len(dfTrain) 
end = len(dfTrain) + len(dfTest) - 1
print(start)
print(end)
  
# predictions for one-year against the dfTest set 
print("\n*** Time Series Model - Predict ***")
lPredict = vResult.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
dfTest['Predict'] = lPredict
print("Done ...")

# predictions for one-year against the dfTest set 
print("\n*** Time Series Model - Forecast ***")
lForecast = vResult.get_forecast(end-start+1)
dfTest['Forecast'] = lForecast.predicted_mean
# default conf_int(alpha=0.05) gives 95% conf int || 0.01 => 99%  0.1 => 90%
dfCI = lForecast.conf_int()
dfCI.columns = ['LoPredInt','UpPredInt']
dfTest = pd.concat([dfTest,dfCI],axis=1)
print("Done ...")

# print
print("\n*** Time Series Model - dfTest ***")
print(dfTest)

# plot
print("\n*** Plot Time Series - dfTest ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=dfTrain.index, y=vColData, data=dfTrain, label='dfTrain', color='k')
sns.lineplot(x=dfTest.index, y=vColData, data=dfTest, label='dfTest.Actual', color='b')
sns.lineplot(x=dfTest.index, y="Forecast", data=dfTest, label='dfTest.Forecast', color='r')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle + " - Plot Time Series - dfTest")
plt.show()

# calculate mean squared error 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error 
mse = mean_squared_error(dfTest[vColData], dfTest["Forecast"]) 
print(mse)

# calculate Root mean squared error 
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print (rmse)

# scatter index
# scatter index less than 1; the predictions are good
print('\n*** Scatter Index ***')
si = rmse/dfTest[vColData].mean()
print(si)

# future date list
print("\n*** Dates For New Forecast ***")
lsPred =  tsutils.create_future_date_list(dfTest.index[-1],n=12,p='Months')
for vPred in lsPred:
    print(vPred)

# convert list to dataframe
print("\n*** Dataframe For New Forecast ***")
dfPred = pd.DataFrame(lsPred,columns=[vColDate])
dfPred[vColData] = None
print(dfPred.head())

# set the 'Month' as index
print("\n*** Dataframe For New Forecast - Again ***")
dfPred.set_index(vColDate, inplace=True)
print(dfPred.head())

# prepare start end to match with dfTest + dfPred
# required is period serial number 
print("\n*** Time Series Model - Period Serial Nos ***")
start = len(dfTrain) 
end = len(dfTrain) + len(dfTest) + len(dfPred) - 1
print(start)
print(end)

# predictions for one-year against the dfTest set 
print("\n*** Time Series Model - Predict ***")
lPredict = vResult.predict(start, end, 
                              typ = 'levels').rename("Predictions") 
dfPred['Predict'] = lPredict
print("Done ...")

# predictions for one-year against the dfPred set 
print("\n*** Time Series Model - Forecast ***")
lForecast = vResult.get_forecast(end-start+1)
dfPred['Forecast'] = lForecast.predicted_mean[-len(dfPred):]
dfCI = lForecast.conf_int()[-len(dfPred):]
dfCI.columns = ['LoPredInt','UpPredInt']
dfPred = pd.concat([dfPred,dfCI],axis=1)
print(dfPred.head())
print("Done ...")

# plot
print("\n*** Plot dfTest Time Series ***")
plt.figure(figsize=(9,4))
sns.lineplot(x=dfTrain.index, y=vColData, data=dfTrain, label='dfTrain', color='k')
sns.lineplot(x=dfTest.index, y=vColData, data=dfTest, label='dfTest.Actual', color='b')
sns.lineplot(x=dfTest.index, y="Forecast", data=dfTest, label='dfTest.Forecast', color='r')
sns.lineplot(x=dfPred.index, y="Forecast", data=dfPred, label='dfPred.Forecast', color='m')
plt.xticks(rotation=60)
plt.legend()
plt.title(vTitle)
plt.show()
