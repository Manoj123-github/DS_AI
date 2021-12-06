

# imports
# warnings
import warnings
warnings.filterwarnings('ignore')
# pandas
import pandas as pd
# numpy
import numpy as np
# datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta

# more date / time functions
# https://towardsdatascience.com/20-simple-yet-powerful-features-for-time-series-using-date-and-time-af9da649e5dc

###################################################################
# time stamp
###################################################################

# define date var
print('\n*** Date Variable Define ***')
vTestDate = pd.Timestamp('2018-01-01')
print(vTestDate)
print(type(vTestDate))

# can also be used to check / compare
print('\n*** Date Variable Compare ***')
vTestBool = (vTestDate == pd.Timestamp('2018-01-01'))
print(vTestBool)

# define timestamp var
print('\n*** Date Time Variable Define ***')
vTestDate = pd.Timestamp('2018-01-01 10:10:10')
print(vTestDate)
print(type(vTestDate))
 
# current date time
print('\n*** Current Date Time ***') 
# using today
print("Current Date Using Today :", datetime.today())
# using now
print("Current Date Using Now   :", datetime.now())

# current date time via variable
print('\n*** Current Date Time ***') 
# using today
vTestDate = datetime.today()
print("Current Date Using Today :", vTestDate)
print(type(vTestDate))
# using now
vTestDate = pd.Timestamp(datetime.today())
print("Current Date Using Now   :", vTestDate)
print(type(vTestDate))

# print 
print('\n*** Date Time Extracts ***') 
print("Date   : ", vTestDate)
print("Year   : ", vTestDate.year)
print("Month  : ", vTestDate.month)
print("Day    : ", vTestDate.day)
print("Hour   : ", vTestDate.hour)
print("Minute : ", vTestDate.minute)
print("Second : ", vTestDate.second)
print("DoW    : ", vTestDate.weekday())
print("DoWN   : N/A")
print("DoY    : ", vTestDate.dayofyear)
print("DiM:"  " ", print(vTestDate.daysinmonth))


# string from date
# url for reference http://strftime.org
print('\n*** String Fromat From Date ***') 
print("dd-mm-yy         : ", vTestDate.strftime('%d-%m-%y'))
print("dd-mm-yyyy       : ", vTestDate.strftime('%d-%m-%Y'))
print("dd-mmm-yyyy      : ", vTestDate.strftime('%d-%b-%Y'))
print("dd-mmmm-yyyy     : ", vTestDate.strftime('%d-%B-%Y'))

# string from date time
# url for reference http://strftime.org
print('\n*** String Fromat From Date Time ***') 
print("dd-mmm-yyyy hh:mm              : ", vTestDate.strftime('%d-%b-%Y %H:%M'))
print("dd-mmm-yyyy hh:mm:ss           : ", vTestDate.strftime('%d-%b-%Y %H:%M:%S'))
print("DOW dd-mmm-yyyy hh:mm:ss       : ", vTestDate.strftime('%a %d-%b-%Y %H:%M:%S'))
print("dd-mmm-yyyy hh:mm:ss ap        : ", vTestDate.strftime('%d-%b-%Y %H:%M:%S %p'))
print("DOW dd-mmm-yyyy hh:mm:ss ap    : ", vTestDate.strftime('%a %d-%b-%Y %H:%M:%S %p'))
print("DowDow dd-mmm-yyyy hh:mm:ss ap : ", vTestDate.strftime('%A %d-%b-%Y %H:%M:%S %p'))

# date maths
print('\n*** Date Maths - Addition ***') 
# add 2 years  
vNewDate = vTestDate + relativedelta(years=2)
print("Add Years")
print(vTestDate)  
print(vNewDate)  
# add 6 months
vNewDate = vTestDate + relativedelta(months=6)
print("Add Months")
print(vTestDate)  
print(vNewDate)  
# add 1 day
vNewDate = vTestDate + relativedelta(days=1)
print("Add Days")
print(vTestDate)  
print(vNewDate)  
# add 6 hours
vNewDate = vTestDate + relativedelta(hours=6)
print("Add Hours")
print(vTestDate)  
print(vNewDate)  
# add 1 mins
vNewDate = vTestDate + relativedelta(minutes=1)
print("Add Mins")
print(vTestDate)  
print(vNewDate)  
# add 2 secs  
vNewDate = vTestDate + relativedelta(seconds=2)
print("Add Secs")
print(vTestDate)  
print(vNewDate)  

# date maths
print('\n*** Date Maths - Subtraction ***') 
# subtract 2 years  
vNewDate = vTestDate - relativedelta(years=2)
print("Sub Years")
print(vTestDate)  
print(vNewDate)  
# subtract 6 months
vNewDate = vTestDate - relativedelta(months=6)
print("Sub Months")
print(vTestDate)  
print(vNewDate)  
# subtract 1 day
vNewDate = vTestDate - relativedelta(days=1)
print("Sub Days")
print(vTestDate)  
print(vNewDate)  
# subtract 6 hours
vNewDate = vTestDate - relativedelta(hours=6)
print("Sub Hours")
print(vTestDate)  
print(vNewDate)  
# subtract 1 mins
vNewDate = vTestDate - relativedelta(minutes=1)
print("Sub Mins")
print(vTestDate)  
print(vNewDate)  
# subtract 2 secs  
vNewDate = vTestDate - relativedelta(seconds=2)
print("Sub Secs")
print(vTestDate)  
print(vNewDate)  

# date maths
print('\n*** Date Maths - Multiple Params***') 
# pass multiple parameters (1 day and 5 minutes)  
vNewDate = vTestDate + relativedelta(days=1,minutes=5)  
print("New Date")
print(vTestDate)  
print(vNewDate)  

# date maths - diff in days
print('\n*** Date Maths - Diff In Days ***') 
# define timestamp var
vDate1 = pd.Timestamp('2018-01-01')
vDate2 = pd.Timestamp('2018-01-04')
vDelta = vDate2 - vDate1
print(vDate1)
print(vDate2)
print(type(vDelta))
print(vDelta)
print(vDelta.days)
# define timestamp var
vDate1 = pd.Timestamp('2018-01-01')
vDate2 = pd.Timestamp('2018-02-01')
vDelta = vDate2 - vDate1
print(vDate1)
print(vDate2)
print(type(vDelta))
print(vDelta)
print(vDelta.days)
# define timestamp var
vDate1 = pd.Timestamp('2018-01-01')
vDate2 = pd.Timestamp('2018-03-01')
vDelta = vDate2 - vDate1
print(vDate1)
print(vDate2)
print(type(vDelta))
print(vDelta)
print(vDelta.days)
# define timestamp var
vDate1 = pd.Timestamp('2016-01-01')
vDate2 = pd.Timestamp('2018-03-01')
vDelta = vDate2 - vDate1
print(vDate1)
print(vDate2)
print(type(vDelta))
print(vDelta)
print(vDelta.days)

###################################################################
# date periods
###################################################################

# date periods
print('\n*** Define Periods ***') 

# define period - annual
vTestPerd = pd.Period('2018')
print(vTestPerd)
print(type(vTestPerd))
vTestPerd

# define period - annual
vTestPerd = pd.Period('2018-01', 'A')
print(vTestPerd)
print(type(vTestPerd))
vTestPerd

# define period - annual
vTestPerd = pd.Period('2018-04', freq='A-MAR')
print(vTestPerd)
print(type(vTestPerd))
vTestPerd

# define period - monthly
vTestPerd = pd.Period('2018-01')
print(vTestPerd)
print(type(vTestPerd))
vTestPerd

# define period - monthly
vTestPerd = pd.Period('2018-01-01', 'M')
print(vTestPerd)
print(type(vTestPerd))
vTestPerd

# define period - quarterly
vTestPerd = pd.Period('2018-01-01', 'Q')
print(vTestPerd)
print(type(vTestPerd))
vTestPerd

# define period - daily
vTestPerd = pd.Period('2018-01-01')
print(vTestPerd)
print(type(vTestPerd))
vTestPerd

# define period - daily
vTestPerd = pd.Period('2018-01', 'D')
print(vTestPerd)
print(type(vTestPerd))
vTestPerd

# convert annual to monthly
vTestPerd = pd.Period('2018')
print(vTestPerd)
vNewPerd = vTestPerd.asfreq("M")
print(vNewPerd)

# convert annual to daily
vTestPerd = pd.Period('2018')
print(vTestPerd)
vNewPerd = vTestPerd.asfreq("D")
print(vNewPerd)

# convert month to daily
vTestPerd = pd.Period('2018-02')
vNewPerd = vTestPerd.asfreq("D")
print(vTestPerd)
print(vNewPerd)

# convert month to annual
vTestPerd = pd.Period('2018-01')
vNewPerd = vTestPerd.asfreq("A")
print(vTestPerd)
print(vNewPerd)

# convert daily to monthly
vTestPerd = pd.Period('2018-01-21')
vNewPerd = vTestPerd.asfreq("M")
print(vTestPerd)
print(vNewPerd)

# convert daily to annual
vTestPerd = pd.Period('2018-01-21')
vNewPerd = vTestPerd.asfreq("A")
print(vTestPerd)
print(vNewPerd)

# period maths - daily
vTestPerd = pd.Period('2018-01-21')
vNewPerd = vTestPerd + 1
print(vTestPerd)
print(vNewPerd)

###################################################################
# date periods maths
###################################################################

# date periods maths
print('\n*** Date Periods Maths ***') 

# period maths - months
vTestPerd = pd.Period('2018-01')
vNewPerd = vTestPerd + 1
print(vTestPerd)
print(vNewPerd)
vTestPerd

# period maths - year
vTestPerd = pd.Period('2018')
vNewPerd = vTestPerd + 1
print(vTestPerd)
print(vNewPerd)
vTestPerd

###################################################################
# date range / sequences
###################################################################

# date range / sequences
print('\n*** Date Range / Sequence ***') 

# set timestamp range
print("Range: start ='2018-01-01', end='2018-12-31', freq='D'")
index = pd.date_range(start ='2018-01-01', end='2018-12-31', freq='D')
print(index)
print(type(index))

# set timestamp range
print("Range: start ='2018-01-01', end='2018-12-31', freq='M'")
index = pd.date_range(start ='2018-01-01', end='2018-12-31', freq='M')
print(index)
print(type(index))

# set timestamp range
index = pd.date_range(start ='2018-01-01', end='2020-12-31', freq='A')
print("Range: start ='2018-01-01', end='2020-12-31', freq='A'") 
print(index)
print(type(index))

# convert to period
print("Range: start ='2018-01-01', end='2018-12-31', freq='M' As Period")
index = pd.date_range(start ='2018-01-01', end='2018-12-31', freq='M')
index = index.to_period()
print(index)
print(type(index))

###################################################################
# time series dataframe - read data
###################################################################

# read data
print('\n*** Read Data ***') 
df = pd.read_csv("./data/air-passengers.csv")
print(df.info())
print(df.head())

# covert to datetime
print("\n*** Convert Object To Date ***")
df['Month'] = pd.to_datetime(df['Month'],format="%Y-%m")
print(df.info())
print(df.head())

# set the 'Month' as index
print('\n*** Set Index ***') 
df = df.set_index('Month')
print(df.info())
print(df.head())

###################################################################
# time series dataframe - read data one more
###################################################################

# read data
print('\n*** Read Data ***') 
df = pd.read_csv("./data/synth-data.csv")
print(df.info())
print(df.head())

# covert to datetime
print("\n*** Convert Object To Date ***")
df['date'] = pd.to_datetime(df['date'],format="%Y-%m-%d")
print(df.info())
print(df.head())

# set the 'Month' as index
print('\n*** Set Index ***') 
df = df.set_index('date')
print(df.info())
print(df.head())
