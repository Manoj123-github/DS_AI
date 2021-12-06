
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

# plott decomposed series
def plot_decomposed_series(pRes):
    """
    returns: 
        nil but plots decomposed series
    usage: 
        plot_decomposed_series(pRes)
	param:
		decomposition result
    """ 
    print("\n*** Decomposed Series***")
    plt.figure(figsize=(8,8))
    pRes.plot()
    plt.show()
        
# visualize individual results - trend
def plot_decomposed_trend(pRes):
    """
    returns: 
        nil but plots trend from decomposed series
    usage: 
        plot_decomposed_trend(pRes)
	param:
		decomposition result
	"""
    print("*** Graph - Trend ***")
    plt.figure(figsize=(10,5))
    plt.plot(pRes.trend, label='trend', color='b')
    plt.title("Trend Graph")
    plt.xlabel('Year')
    plt.legend(loc='upper right')
    plt.show()
        
# visualize individual results - season
def plot_decomposed_season(pRes):
    """
    returns: 
        nil but plots season from decomposed series
    usage: 
        plot_decomposed_season(pRes)
	param:
		decomposition result
	"""
    print("*** Graph - Seasonality ***")
    plt.figure(figsize=(10,5))
    plt.plot(pRes.seasonal, label='seasonal', color='b')
    plt.title("Seasonality Graph")
    plt.xlabel('Year')
    plt.legend(loc='upper right')
    plt.show()
        
# visualize individual results - resid
def plot_decomposed_residual(pRes):
    """
    returns: 
        nil but plots residual from decomposed series
    usage: 
        plot_decomposed_residual(pRes)
	param:
		decomposition result
	"""
    print("*** Graph - Residual ***")
    plt.figure(figsize=(10,5))
    plt.plot(pRes.resid, label='residual', color='b')
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
def plot_decomposed_runseq(pRes):
    """
    returns: 
        nil but plots run sequence of decomposed series
    usage: 
        plot_decomposed_runseq(pRes)
	param:
		decomposition result
	"""
    print("*** Graph - Run Sequence ***")
    plt.figure(figsize=(10,5))
    plt.plot(pRes.trend, label='trend', color='b')
    plt.plot(pRes.seasonal, label='seasonal', color='g')
    plt.plot(pRes.resid, label='residual', color='k')
    plt.title("Run Sequence Graph")
    plt.xlabel('Year')
    plt.legend(loc='upper right')
    plt.show()


# augmented dickey-fuller test
# this is one of the statistical tests for checking stationarity. 
# Here the null hypothesis is that the TS is non-stationary. 
# The test results comprise of a Test Statistic and some Critical Values for 
# difference confidence levels. 
# if p-value < 0.05, we can reject the 
# null hypothesis and say that the series is stationary.  
# If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the 
# null hypothesis and say that the series is stationary. 
from statsmodels.tsa.stattools import adfuller
# augmented dickey-fuller test
def adfTest(ts):
    """
    returns: 
        adf test results
		states if ts is stationary or not stationary
    usage: 
        adfTest(ts)
	param:
		python time series format (period as index & single col of data)
	"""
    # adf test
    adfResult = adfuller(ts, autolag='AIC')
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
    return (adfRetValue)

        
# KPSS Test
# The KPSS test, on the other hand, is used to test for trend stationarity. 
# The null hypothesis and the P-Value interpretation is just the opposite 
# of ADF test. The below code implements these two tests using statsmodels 
# package in python.
from statsmodels.tsa.stattools import kpss
# kpss test
def kpssTest(ts):
    """
    returns: 
        kpss test results
		states if ts is stationary or not stationary
    usage: 
        kpssTest(ts)
	param:
		python time series format (period as index & single col of data)
	"""
    # kpss test
    kpsResult = kpss(ts.values, regression='c')
    # kpss test result as data series
    kpsRetValue = pd.Series(kpsResult[0:3], index=['Test Statistic','p-value','#Observations Used'])
    # kpss test critical value add to data series
    for key,value in kpsResult[3].items():
        kpsRetValue['Critical Value (%s)'%key] = value
    #print(kpsRetValue)
    # p-value test
    pvResult = 'Time Series is ' + ('NOT Stationary' if kpsRetValue[1] < 0.05 else  'Stationary')
    kpsRetValue['P-Value Test'] = pvResult
    # t-stats test
    kpsResult[0] > kpsResult[3]['5%']
    tsResult = 'Time Series is ' + ('NOT Stationary' if kpsResult[0] > kpsResult[3]['5%'] else 'Stationary')
    kpsRetValue['T-Stats Test'] = tsResult
    return (kpsRetValue)

"""
Auto Correlation Function (ACF) plot
ACF is an (complete) auto-correlation function which gives us values of 
auto-correlation of any series with its lagged values. We plot these values 
along with the confidence band and tada! We have an ACF plot. 
In simple terms, it describes how well the present value of the series is 
related with its past values. A time series can have components like trend, 
seasonality, cyclic and residual. ACF considers all these components while 
finding correlations hence it’s a ‘complete auto-correlation plot’.   
Use ACF plot for most optimal in the MA(q) model    
q is the lag value at which ACF plot crosses the upper confidence interval 
for the first time. These q lags will act as our features while forecasting 
the MA time series.
"""
# acf plot rich
def plot_acf(df, pColDate, pColData):
    """
    returns: 
        nil but plots Auto Correlation Function (ACF)
        this is a rich plot
    usage: 
        plot_acf(df, pColDate, pColData)
	param:
		ts, DateCol, DataCol
	"""
    from statsmodels.graphics.tsaplots import plot_acf
    plt.rcParams['figure.figsize'] = (8,5)
    plt.figure()
    plt.ylim(-2, 2)
    plot_acf(df.values.tolist(), lags=50)
    plt.axhline(y=0,linestyle='--',color='red')
    #plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',colsimpleor='red')
    plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='red')
    plt.title("Auto Corelation Plot")
    plt.xlabel('Lags')
    plt.show()

# plot pacf line
def plot_acf_line(df, pColDate, pColData):
    """
    returns: 
        nil but plots Auto Correlation Function (ACF)
        this is a simple line plot
    usage: 
        plot_acf_line(df, pColDate, pColData)
	param:
		ts, DateCol, DataCol
	"""
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
# pacf plot rich
def plot_pacf(df, pColDate, pColData):
    """
    returns: 
        nil but plots Partial Auto Correlation Function (PACF)
        this is a rich plot
    usage: 
        plot_pacf(df, pColDate, pColData)
	param:
		ts, DateCol, DataCol
	"""
    from statsmodels.graphics.tsaplots import plot_pacf
    plt.rcParams['figure.figsize'] = (8,5)
    plt.figure(figsize=(5,5))
    #plt.ylim(-2, 2)
    plot_pacf(df.values.tolist(), lags=50)
    plt.axhline(y=0,linestyle='--',color='red')
    #plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='red')
    plt.title("Partial Auto Corelation Plot")
    plt.xlabel('Lags')
    plt.show()
    
# pacf plot line
def plot_pacf_line(df, pColDate, pColData):
    """
    returns: 
        nil but plots Partial Auto Correlation Function (PACF)
        this is a rich plot
    usage: 
        plot_pacf_line(df, pColDate, pColData)
	param:
		ts, DateCol, DataCol
	"""
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

# create future datelist
def create_future_date_list(pStartDate, n, p):
    """
    returns: 
        list of future dates from StartData based on n & p
    usage: 
        create_future_date_list(pStartDate, n, p)        
	param:
		pStartDate, n, p
        n = 12          # number of dates
        p = 'Months'    # Days / Months / Years
	"""
    from dateutil.relativedelta import relativedelta
    vStartDate = pStartDate
    lstDateList = []
    #lstDateList.append(vStartDate)
    for i in range(1,n+1):
        if p == 'Days':
            vNewDate = vStartDate + relativedelta(days=i) 
        if p == 'Months':
            vNewDate = vStartDate + relativedelta(months=i) 
        if p == 'Years':
            vNewDate = vStartDate + relativedelta(years=i) 
        lstDateList.append(vNewDate)
    return (lstDateList)

