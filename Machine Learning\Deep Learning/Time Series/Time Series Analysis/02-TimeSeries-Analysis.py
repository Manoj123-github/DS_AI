

# https://www.machinelearningplus.com/time-series/time-series-analysis-python/
# https://otexts.com/fpp2/autocorrelation.html

"""
Time Series Components
Level Trend Seasonality & Noise
A series is thought to be an aggregate or combination of these four components.
All series have a level and noise. The trend and seasonality components are optional.
It is helpful to think of the components as combining either additively or multiplicatively.

Additive Model
An additive model suggests that the components are added together as follows:
y(t) = Level + Trend + Seasonality + Noise
An additive model is linear where changes over time are consistently made by the same amount.
A linear trend is a straight line.
A linear seasonality has the same frequency (width of cycles) and amplitude (height of cycles).

Multiplicative Model
A multiplicative model suggests that the components are multiplied together as follows:
y(t) = Level * Trend * Seasonality * Noise
A multiplicative model is nonlinear, such as quadratic or exponential. Changes increase or decrease over time.
A nonlinear trend is a curved line.
A non-linear seasonality has an increasing or decreasing frequency and/or amplitude over time.

Analysis
We will go over how to import time series in python into a pandas dataframe. 
We will then inspect the dataframe for missing values, change the column names 
if necessary, convert the date column to datetime, and set the index for the 
dataframes. We will then move on to provide the descriptive (summary) statistics, 
plot the time series and the plot the components of time series

Datasets:
Air Passengers - 
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
#%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8, 8)
# seaborn
import seaborn as sns


################################################################
# analyse time series
################################################################
   
# read data
df = pd.read_csv("./data/air-passengers.csv")
pTitle = "Air Passengers"
pColDate = "Month"
pColData  = "Passengers"
pDatFrmt  = "%Y-%m"  

print("\n*** Input ***")
print("Title:",pTitle)
print("Month:",pColDate)
print("Data :",pColData)
print("Frmt :",pDatFrmt)

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
if df[pColData].isnull().sum() > 0:
    df[pColData] = df[pColData].interpolate(method ='linear')
print(df.isnull().sum()) 

# covert to datetime
df[pColDate] = pd.to_datetime(df[pColDate],format=pDatFrmt)
print("\n*** Structure Again ***")
print(df.info())

# set the 'Month' as index
df.set_index(pColDate, inplace=True)
print("\n*** Head Again ***")
df.head()

# plot
print("\n*** Plot Time Series ***")
plt.figure(figsize=(10,5))
sns.lineplot(x=df.index, y=pColData, data=df, color='b')
plt.xticks(rotation=60)
plt.title(pTitle)
plt.show()

# seasonal decompose - model additive 
import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(df, model='additive')
print("\n*** Decomposed Series***")
plt.figure(figsize=(8,8))
res.plot()
plt.show()

# # seasonal decompose - model multiplicative
# import statsmodels.api as sm
# res = sm.tsa.seasonal_decompose(df,model='multiplicative')
# print("\n*** Decomposed Series***")
# plt.figure(figsize=(8,8))
# res.plot()
# plt.show()

# visualize individual results - trend
print("*** Graph - Trend ***")
plt.figure(figsize=(10,5))
plt.plot(res.trend, label='trend', color='b')
plt.title("Trend Graph")
plt.xlabel('Year')
plt.legend(loc='upper right')
plt.show()

# visualize individual results - seasonal
print("*** Graph - Seasonality ***")
plt.figure(figsize=(10,5))
plt.plot(res.seasonal, label='seasonal', color='b')
plt.title("Seasonality Graph")
plt.xlabel('Year')
plt.legend(loc='upper right')
plt.show()

# visualize individual results - resid
print("*** Graph - Residual ***")
plt.figure(figsize=(10,5))
plt.plot(res.resid, label='residual', color='b')
plt.title("Residual Graph")
plt.xlabel('Year')
plt.legend(loc='upper right')
plt.show()

"""
Series Additive or Multiplicative - decide viewing run sequence graph
If the seasonality and residual components are independent of the trend, then 
you have an additive series. 
If the seasonality and residual components are in fact dependent, meaning they 
fluctuate on trend, then you have a multiplicative series.
"""
# visualize individual results - run sequence
print("*** Graph - Run Sequence ***")
plt.figure(figsize=(10,5))
plt.plot(res.trend, label='trend', color='b')
plt.plot(res.seasonal, label='seasonal', color='g')
plt.plot(res.resid, label='residual', color='k')
plt.title("Run Sequence Graph")
plt.xlabel('Year')
plt.legend(loc='upper right')
plt.show()


"""
Stationarity
Stationarity is a property of a time series. A stationary series is one 
where the values of the series is not a function of time.
That is, the statistical properties of the series like mean, variance and 
autocorrelation are constant over time. 
A stationary time series is devoid of seasonal effects as well.
We can use either of the two test
-- Augmented Dickey-Fuller Test
-- KPSS Test
"""

"""
Augmented Dickey-Fuller Test
this is one of the statistical tests for checking stationarity. 
Here the null hypothesis is that the TS is non-stationary. 
The test results comprise of a Test Statistic and some Critical Values for 
difference confidence levels. 
if p-value < 0.05, we can reject the 
null hypothesis and say that the series is stationary.  
If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the 
null hypothesis and say that the series is stationary. 
"""
print('\n*** Augmented Dickey-Fuller Test ***')
from statsmodels.tsa.stattools import adfuller
# adf test
adfResult = adfuller(df, autolag='AIC')
# adf test result as data series
adfRetValue = pd.Series(adfResult[0:4], index=['Test Statistic','p-value','#Lags Used','#Observations Used'])
# adf test critical value add to data series
for key,value in adfResult[4].items():
    adfRetValue['Critical Value (%s)'%key] = value
#print(kpsRetValue)
# p-value test
pvResult = 'Time Series is ' + ('Stationary' if adfRetValue[1] < 0.05 else  'NOT Stationary')
adfRetValue['P-Value Test'] = pvResult
# t-stats test
tsResult = 'Time Series is ' + ('Stationary' if adfResult[0] < adfResult[4]['5%'] else 'NOT Stationary')
adfRetValue['T-Stats Test'] = tsResult
print(adfRetValue)

"""
KPSS Test
The KPSS test, on the other hand, is used to test for trend stationarity. 
The null hypothesis and the P-Value interpretation is just the opposite 
of ADF test. The below code implements the KPSS test using statsmodels 
package in python.
"""
print('\n*** KPSS Test ***')
#import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import kpss
# kpss test
kpsResult = kpss(df.values, regression='c')
# kpss test result as data series
kpsRetValue = pd.Series(kpsResult[0:3], index=['Test Statistic','p-value','#Observations Used'])
# kpss test critical value add to data series
for key,value in kpsResult[3].items():
    kpsRetValue['Critical Value (%s)'%key] = value
    # print(kpsRetValue)
# p-value test
pvResult = 'Time Series is ' + ('NOT Stationary' if kpsRetValue[1] < 0.05 else  'Stationary')
kpsRetValue['P-Value Test'] = pvResult
# t-stats test
kpsResult[0] > kpsResult[3]['5%']
tsResult = 'Time Series is ' + ('NOT Stationary' if kpsResult[0] > kpsResult[3]['5%'] else 'Stationary')
kpsRetValue['T-Stats Test'] = tsResult
print(kpsRetValue)


"""
Complete Auto Correlation Function (ACF) plot
ACF is an (complete) auto-correlation function which gives us values of 
auto-correlation of any series with its lagged values. We plot these values 
along with the confidence band and we have an ACF plot. 
In simple terms, it describes how well the present value of the series is 
related with its past values. A time series can have components like trend, 
seasonality, cyclic and residual. ACF considers all these components while 
finding correlations hence it’s a ‘complete auto-correlation plot’.    

Use ACF plot for most optimal in the MA(q) model    
q is the lag value at which ACF plot crosses the upper confidence interval 
for the first time. These q lags will act as our features while forecasting 
the MA time series.
"""
# plot acf - fancy
print('\n*** Complete ACF Plot ***')
from statsmodels.graphics.tsaplots import plot_acf
plt.rcParams['figure.figsize'] = (8,5)
plt.figure()
plot_acf(df.values.tolist(), lags=50)
plt.axhline(y=0,linestyle='--',color='red')
#plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',colsimpleor='red')
plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='red')
plt.title("Auto Corelation Plot")
plt.xlabel('Lags')
plt.show()

# plot pacf - simple
print('\n*** Complete ACF Plot ***')
from statsmodels.tsa.stattools import acf
acf_50 = acf(df[pColData], nlags=50)
plt.rcParams['figure.figsize'] = (8,5)
plt.figure()
plt.ylim(-2, 2)
plt.plot(acf_50, color='b')
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='gray')
plt.title('Auto Correlation Function')
plt.show


"""
Partial Auto Correlation Function (PACF) plot
PACF is a partial auto-correlation function. Basically instead of finding 
correlations of present with lags like ACF, it finds correlation of the 
residuals (which remains after removing the effects which are already 
explained by the earlier lag(s)) with the next lag value hence ‘partial’ 
and not ‘complete’ as we remove already found variations before we find 
the next correlation. So if there is any hidden information in the residual 
which can be modeled by the next lag, we might get a good correlation and 
we will keep that next lag as a feature while modeling. 
While modeling we don’t want to keep too many features which are correlated 
as that can create multicollinearity issues. Hence we need to retain only 
the relevant features.
Use PACF plot for most optimal in the AR(p) model.
p is the lag value at which PACF plot crosses the upper confidence interval 
for the first time. These p lags will act as our features while forecasting 
the AR time series.
"""

# pacf plot fancy
print('\n*** Partial ACF Plot ***')
from statsmodels.graphics.tsaplots import plot_pacf
plt.rcParams['figure.figsize'] = (8,5)
plt.figure(figsize=(5,5))
plot_pacf(df.values.tolist(), lags=50)
plt.axhline(y=0,linestyle='--',color='red')
#plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='red')
plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='red')
plt.title("Partial Auto Corelation Plot")
plt.xlabel('Lags')
plt.show()

# pacf plot simple
print('\n*** Partial ACF Plot ***')
from statsmodels.tsa.stattools import pacf
pacf_50 = pacf(df[pColData], nlags=50)
plt.rcParams['figure.figsize'] = (8,5)
plt.figure()
plt.ylim(-2, 2)
plt.plot(pacf_50,color='b')
plt.axhline(y=0,linestyle='--',color='red')
plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='red')
plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='red')
plt.title('Partial Autocorrelation Function')
plt.show
