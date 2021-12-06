# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:31:52 2020

@author: Manoj Yadav
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Python/data/Data visulation/nifty-data.csv')
print(df.info())
print(df.head())
print(df.columns)
# Exercise 1:
# Show 'Symbol','AvgPrice','ClosePrice' in Bargraph of first 10 rec
# Compute AvgPrice = TradeValue / TradeQty

df['AvgPrice'] = df['TradeValue'] / df['TradeQty']
print(df.head())

dfn = df[0:9]
dfn = dfn[['Symbol','AvgPrice','ClosePrice']]
plt.figure()
sns.barplot(x="Symbol", y="AvgPrice", data=dfn,
            label="ClosePricee", color="b")
plt.xticks(rotation=45)
plt.legend()
plt.show()



# Exercise 2:
# create histogram of 'AvgPrice' for Symbol = "ACC"
# Compute AvgPrice = TradeValue / TradeQty
df['AvgPrice'] = df['TradeValue'] /df['TradeQty']
print(df.head())
df = df[df['Symbol'] == 'ACC']
plt.figure()
#sns.distplot(df.AvgPrice, bins=7, color='r')
sns.distplot(df.AvgPrice, hist=False, bins=7, color='r', vertical=False)
#sns.distplot(dfn.AvgPrice, kde=False, bins=7, color='r')
plt.show()


# Exercise 3:
# create boxplot of 'OpenPrice','HighPrice','LowPrice','ClosePrice','AvgPrice' for Symbol = "ACC"
# Compute AvgPrice = TradeValue / TradeQty
df['AvgPrice'] = df['TradeValue'] - df['TradeQty']
print(df.head())
tdf = tdf[tdf['Symbol'] == 'ACC']
tdf.head()
tdf = df.drop(['Mkt', 'Series', 'Sector','NameOfTheSecurityInNse', 'PrevClose','TradeValue', 'TradeQty','IndSec', 'CoprInd',
       'Trades', '52W_High', '52W_Low'], axis=1)
tdf.head()

plt.figure()
sns.boxplot(tdf)
plt.show()



# Exercise 4
# Line Graph showing ClosePrice & RecDate of all TATA Companies

dfn = df[ (df['Symbol'] == 'TATAMOTERS') | (df['Symbol'] == 'TATA MOTORS LIMITED') | (df['Symbol'] == 'TATAPOWER') 
         | (df['Symbol'] == 'TATA POWER CO LTD')| (df['Symbol'] == 'TATASTEEL')| (df['Symbol'] == 'TATA STEEL LIMITED') 
         | (df['Symbol'] == 'TATA CONSULTANCY SERV LT')]
print(dfn[['RecDate','Symbol','ClosePrice'] ])          
dfn.head()
plt.figure()
sns.pointplot(x="RecDate", y="ClosePrice", data=dfn, color="b", scale=0.5)
plt.xticks(rotation=60)
ax = plt.axes()
plt.show()

# Exercise 5
# - Compute AvgTradePrice
# - Compute Group (on Symbol) Average of AvgTradePrice 
# - Bar Graph showing Symbol & Average of AvgTradePrice of all TATA Companies

dfn=df[df["NameOfTheSecurityInNse"].str.contains("TATA")].groupby(['Symbol'])['AvgPrice'].mean().reset_index()
dfn=dfn.rename(columns={"AvgPrice":"Average of AvgTradePrice"})
plt.figure()
sns.barplot(data=dfn, x="Symbol", y="Average of AvgTradePrice", color='b')
plt.xticks(rotation=60)
plt.legend()
plt.show()



# Exercise 6
# - Bar Graph showing Openprice, HighPrice, LowPrice, ClosePrice of all Mahindra Companies
dfnew=df[df["NameOfTheSecurityInNse"].str.contains("MAHINDRA")]
dfnew=dfn[['NameOfTheSecurityInNse','OpenPrice','HighPrice','LowPrice','ClosePrice']].rename(columns={"NameOfTheSecurityInNse":"NameofCompany"})
dfnew=pd.melt(dfn, id_vars=["NameofCompany"])
dfnew.columns = ["NameofCompany","PType","Price"]
plt.figure()        
sns.barplot(data=dfn, x="NameofCompany", y="Price", hue="PType")
plt.xticks(rotation=60)
plt.legend()
plt.show()