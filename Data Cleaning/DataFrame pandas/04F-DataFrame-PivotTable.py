
# imports
import pandas as pd
import numpy as np


##########################################################################
#### pivot table with aggregation
##########################################################################

# read dataset

df = pd.read_csv('C:\Python\data\dataframe pandas/northwind-m.csv')
# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())

df['SaleAmt'] = df['Quantity'] * df['SalePrice'] * (1-df['SaleDiscount'])
df['CostAmt'] = df['Quantity'] * df['CostPrice']
df['Profit'] = df['SaleAmt'] - df['CostAmt']

# groupby count
print(df.groupby(['CompanyName', 'ProductName'])['RefID'].count())
# groupby sum
print(df.groupby(['CompanyName', 'ProductName'])['SaleAmt'].sum())


# pivot table - Company Name
# index - single column
pd.pivot_table(df, Quantity=["CompanyName"])
# index - multi column
pd.pivot_table(df, index=["CompanyName","ProductName"])
# value - single
pd.pivot_table(df, index=["CompanyName","ProductName"], values=["SaleAmt"])
# value - multi col
pd.pivot_table(df, index=["CompanyName","ProductName"], values=["SaleAmt","CostAmt","Profit"])
# agg funcs = len np.sum, np.mean, np.max, np.min
pd.pivot_table(df, index=["CompanyName","ProductName"], values=["SaleAmt","CostAmt","Profit"], aggfunc=[np.sum])
# assign to df
dfcn = pd.pivot_table(df, index=["CompanyName","ProductName"], values=["SaleAmt","CostAmt","Profit"], aggfunc=[np.sum])

# pivot table - Product Name
dfpn = pd.pivot_table(df, index=["ProductName","CompanyName"], values=["SaleAmt","CostAmt","Profit"], aggfunc=[np.sum])

# pivot table - Category Name
pd.pivot_table(df, index=["CategoryName"], values=["SaleAmt","CostAmt","Profit"], aggfunc=[np.sum])


##########################################################################
#### pivot table without aggregation into cols
##########################################################################

df = pd.read_csv('./data/iip-data-long.csv')

# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())

# pivot table - Company Name
# index - single column
dfp = pd.pivot_table(df, index=["ItemDescription"], columns=["variable"])

# print
print(dfp)
