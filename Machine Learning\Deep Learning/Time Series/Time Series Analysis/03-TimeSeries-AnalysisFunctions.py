
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
Google trends-- term search count of the word "vacation"
Retail Furniture and Furnishing data in Millions of Dollars
Adjusted Close Stock price data for Bank of America
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
def analyse_time_series(pFilePath, pTitle, pColDate, pColData, pDatFrmt):

    # # air passengers
    # pFilePath = './data/air-passengers.csv'
    # pTitle = "Air Passengers"
    # pColDate = "Month"
    # pColData  = "Passengers"
    # pDatFrmt  = '%Y-%m'

    # # min temp
    # pFilePath = './data/min-temp-melbourne.csv'
    # pTitle = "Minimum Temperature - Melbourne"
    # pColDate = "Date"
    # pColData  = "Temp"
    # pDatFrmt  = '%Y-%m-%d'

    # read data
    df = pd.read_csv(pFilePath)

    # input parameters    
    print("\n*** Input Parameters ***")
    print("Title:",pTitle)
    print("Month:",pColDate)
    print("Data :",pColData)
    print("Frmt :",pDatFrmt)

    # info
    print("\n*** Structure ***")
    print(df.info())
   
    # retain only date & dat cols
    df = df [[pColDate, pColData]]
    
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
    
    # set the pColDate as index
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
    
    # seasonal decompose  
    import statsmodels.api as sm
    try:
        # additive model
        res = sm.tsa.seasonal_decompose(df,model='additive')
        # multiplicative model
        # res = sm.tsa.seasonal_decompose(df,model='multiplicative')
    except:
        # additive model
        res = sm.tsa.seasonal_decompose(df,period=1,model='additive')
        # multiplicative model
        # res = sm.tsa.seasonal_decompose(df,period=1,model='multiplicative')
    
    # plot decomposed data
    import tsutils
    tsutils.plot_decomposed_series(res)    
    tsutils.plot_decomposed_trend(res)    
    tsutils.plot_decomposed_season(res)    
    tsutils.plot_decomposed_residual(res)    
    tsutils.plot_decomposed_runseq(res)    

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

    # Augmented Dickey-Fuller Test
    print('\n*** Augmented Dickey-Fuller Test ***')
    import tsutils
    print(tsutils.adfTest(df))

    # KPSS Test
    print('\n*** KPSS Test ***')
    import tsutils
    print(tsutils.kpssTest(df))

    """
    Use ACF plot for most optimal in the MA(q) model    
    q is the lag value at which ACF plot crosses the upper confidence interval 
    for the first time. These q lags will act as our features while forecasting 
    the MA time series.
    """
    # acf plot
    print('\n*** ACF Plot ***')
    import tsutils
    tsutils.plot_acf(df, pColDate, pColData)
    tsutils.plot_acf_line(df, pColDate, pColData)

    """
    Use PACF plot for most optimal in the AR(p) model.
    p is the lag value at which PACF plot crosses the upper confidence interval 
    for the first time. These p lags will act as our features while forecasting 
    the AR time series.
    """
    # pacf plot 
    print('\n*** PACF Plot ***')
    import tsutils
    tsutils.plot_pacf(df, pColDate, pColData)
    tsutils.plot_pacf_line(df, pColDate, pColData)

    

################################################################
# analyse time series
################################################################

# air passengers
# https://www.kaggle.com/rakannimer/air-passenger-prediction
# Date period range: 1949-01 to 1960-12
analyse_time_series('./data/air-passengers.csv',
    pTitle = "Air Passengers", 
    pColDate = "Month",
    pColData  = "Passengers",
    pDatFrmt  = '%Y-%m')

# Google Trends
# https://trends.google.com/trends/explore?date=all&geo=US&q=vacation, google trends, term search of the word "vacation", count data
# Date period range: January 2004 to October 2019, 15 years, data is monthly
analyse_time_series("./data/vacation.csv",
    pTitle = "Google Vacation Search Trends", 
    pColDate = "Month",
    pColData  = "Vacation",
    pDatFrmt  = '%Y-%m')

# Advance Retail Sales: Furniture and Home Furnishings Stores
# Source  https://fred.stlouisfed.org/series/RSFHFSN
# Units are in Millions of Dollars, not seasonally adjusted, price
# Date period range is 01/01/1992 to 07/01/2019, monthly data
analyse_time_series("./data/furniture.csv",
    pTitle = "Advance Retail - Furniture Sales", 
    pColDate = "Date",
    pColData  = "Sales",
    pDatFrmt  = '%d-%m-%Y')

# Consumer Price Index: All Items in U.S. City Average, All Urban Consumers (CPIAUCSL)
# https://www.minneapolisfed.org/community/financial-and-economic-education/cpi-calculator-information
# Index 1982-1984=100, Seasonally Adjusted
# Period is 1992-01-01 to 2019-07-01, monthly
# Unit is millions of dollars
# Updated Oct. 10, 2019
analyse_time_series("./data/cpi.csv",
    pTitle = "Consumer Price Index", 
    pColDate = "Date",
    pColData  = "Value",
    pDatFrmt  = '%m-%d-%Y')

# air quality
# https://www.rdocumentation.org/packages/datasets/versions/3.6.2/topics/airquality
# adapted and amended from above source
# Date period range is 01-05-2019 to 30-09-2019, daily data
analyse_time_series('./data/airquality.csv',
    pTitle = "Air Quality",
    pColDate = "Date",
    pColData  = "Ozone",
    pDatFrmt  = '%d-%m-%Y')

# Adjusted Close Stock Price data for JPMorgan, source is Yahoo finance
# Date period range is 02-01-1990 to 10-16-2019, daily data
analyse_time_series("./data/jpmorgan.csv",
    pTitle = "J P Morgan - Adj Close Price", 
    pColDate = "Date",
    pColData  = "AdjClose",
    pDatFrmt  = '%m/%d/%y')






