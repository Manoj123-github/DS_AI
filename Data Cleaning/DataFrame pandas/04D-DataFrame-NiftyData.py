

import pandas as pd
import numpy as np

# read data from csv and load data in a dataframe
df = pd.read_csv('C:/Python/data/nifty-data.csv')

# print df
print(df)

# info df
print(df.info())

# sector
print(df['Sector'].describe())
print(df['Sector'].count())
print(df['Sector'].nunique())
print(df.groupby(['Sector'])['Sector'].count())

# symbol
print(df['Symbol'].describe())
print(df['Symbol'].count())
print(df['Symbol'].nunique())
print(df.groupby(['Symbol'])['Symbol'].count())
print(df.groupby(['Symbol'])['ClosePrice'].min())
print(df.groupby(['Symbol'])['ClosePrice'].mean())
print(df.groupby(['Symbol'])['ClosePrice'].max())

# sector = auto
dfn = df[df['Sector'] == 'Auto']
print(dfn.head())

# sector = auto or petro
dfn = df[ (df['Sector'] == 'Auto') | (df['Sector'] == 'Petro') ]
print(dfn.head())

# ClosePrice >= 1000
dfn = df[df['ClosePrice'] >= 1000 ]
print(dfn.head())

# count of rows with ClosePrice >= 1000
var = df[df['ClosePrice'] >= 1000 ]['ClosePrice'].count()
print(var)

# ClosePrice < 1000
dfn = df[df['ClosePrice'] < 1000 ]
print(dfn.head())
var = df[df['ClosePrice'] < 1000 ]['ClosePrice'].count()
print(var)

# GainOrLoss Number
df['GoLN'] = df['ClosePrice'] - df['PrevClose'] 
print(df.head())

# GainOrLoss Label
df['GoLL'] = ""
# selective update
df['GoLL'] = np.where(df['GoLN']>0, 'Gain', df['GoLL'])
df['GoLL'] = np.where(df['GoLN']<0, 'Loss', df['GoLL'])
df['GoLL'] = np.where(df['GoLN']==0,'Zero', df['GoLL'])
print(df.head())

def sortOHLC(pIndex):
    #print(pIndex)
    vOpenPrice = df['OpenPrice'][pIndex]
    vHighPrice = df['HighPrice'][pIndex] 
    vLowPrice  = df['LowPrice'][pIndex] 
    vClosePrice = df['ClosePrice'][pIndex] 
    #print(vOpenPrice)
    #print(vHighPrice)
    #print(vLowPrice)
    #print(vClosePrice)
    lPriceData = [vOpenPrice, vHighPrice, vLowPrice, vClosePrice]
    lPriceData.sort()
    #print(lPriceData)
    return (lPriceData)

sortOHLC(1)

# using for loop
for index,row in df.iterrows():
    print(index)
    print(row)
    
# using for loop
for i,row in df.iterrows():
    print(i)
    print(df['RecDate'][i])
    print(df['Sector'][i])
    print(df['Symbol'][i])
    print("")

# using for loop
df['SortOHLC'] = ""
for i,row in df.iterrows():
    print(i)
    df.SortOHLC[i] = sortOHLC(i)
print(df)
 
# comprehension
df['SortOHLC'] = ""
print(df)
df['SortOHLC'] = [sortOHLC(i) for i in df.index]
print(df)

# average trade price
df['AvgTradePrice'] = df['TradeValue'] / df['TradeQty']
print(df.head())

# count
var = df[df['AvgTradePrice'] >= 1500 ]['AvgTradePrice'].count()
print(var)

# 
dfNew = df [ df['Sector']=="FinSer" ]
print(dfNew)
print(type(dfNew))
#
dfNew = df [ (df['Sector']=="Auto") | (df['Sector']=="IT") ]
print(dfNew)

#
dfNew = df [ df['Symbol']=="ACC" ]
print(dfNew)

#
dfNew = df [ (df['Symbol']=="TCS") | (df['Symbol']=="WIPRO") | (df['Symbol']=="INFY")]
print(dfNew)

#
dfNew = df [ df['Symbol'].str.startswith("BANK") ]
print(dfNew)

#
dfNew = df [ df['Symbol'].str.endswith("BANK") ]
print(dfNew)

#
dfNew = df [ df['Symbol'].str.contains("BANK") ]
print(dfNew)

#
dfNew = df [ df['Symbol'].str.contains("bank", case=False) ]
print(dfNew)

#
dfNew = df [ (df['Symbol']=="INFY") & (df['GoLN']>0)]['Symbol'].count()
print(dfNew)

#
dfNew = df [ df['ClosePrice'] > df['OpenPrice'] ]
print(dfNew)

#
dfNew = df [ df['AvgTradePrice'] > df['ClosePrice'] ]
print(dfNew)

#0
dfNew = df [ df['NameOfTheSecurityInNse'].str.contains("TATA", case=False) ]
print(dfNew)

#
dfNew = df [ df['NameOfTheSecurityInNse'].str.contains("Mahindra", case=False) ]
print(dfNew)

df=df.rename(columns = {'NameOfTheSecurityInNse':'SecurityName'})
df.info()

df = df.drop('IndSec', axis=1)
df = df.drop('CoprInd', axis=1)
df.info()







