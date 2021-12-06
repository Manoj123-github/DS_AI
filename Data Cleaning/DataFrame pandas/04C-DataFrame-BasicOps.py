

import pandas as pd
import numpy as np

# read data from csv and load data in a dataframe
df = pd.read_csv('./data/Catalog.csv')

# print df
print(df)

# info df
print(df.info())

# summary df
print(df.describe())

#==============================================================================
# # variations
# # load a csv with no headers
# df = pd.read_csv('./data/Catalog.csv', header=None)
# # load a csv while specifying column names
# df = pd.read_csv('./data/Catalog.csv', names=['Title', 'Artist', 'Country', 'Company', 'Price', 'Year'])
# # load a csv while specifying "." as missing values
# df = pd.read_csv('./data/Catalog.csv', na_values=['.'])
# # load a csv while skipping the top 3 rows
# df = pd.read_csv('./data/Catalog.csv', skiprows=3)
# # load a csv while interpreting "," in strings around numbers as thousands seperators
# df = pd.read_csv('./data/Catalog.csv', thousands=',')
# 
#==============================================================================

# see the top & bottom rows of the frame
print(df.head())
print(df.tail())

print(df.head(2))
print(df.tail(3))

# display the index
print(df.index)

# display columns
print(df.columns)

# print dataframe with RowNo as index
# df[n:n]
# Note that Pandas uses zero based numbering, so 0 is the first row, 
# 1 is the second row, etc.
# 1:2 is to be read as starting 1 upto but including 2 ... prints row 1 only
print(df)
print(df[1:2])
print(df[1:-1])
print(df[1:])

# print dataframe column using ColName
# df['ColName']
print(df['Title'])

# print dataframe using RowNo & ColName
print(df)
print(df[0:2]['Title'])

# print dataframe using RowNo & ColName
print(df['Title'])
print(df['Title'][0:2])

# print dataframe rows with condition
# df[ df['ColName'] condition ]

# single condition numeric equals
dfnew = df[  df['Year']==1990  ]
print(dfnew)

# single condition numeric other operators
dfnew = df[df['Year'] <= 1990]
print(dfnew)

# single condition numeric other operators
dfnew = df[df['Year'] >= 1990]
print(dfnew)

# single condition numeric not operator
dfnew = df[df['Year'] != 1990]
print(dfnew)

# single condition string equals
dfnew = df[df['Title'] == 'Red']
print(dfnew)

# single condition string NOT equals
dfnew = df[df['Title'] != 'Red']
print(dfnew)

# single condition string less than equals
dfnew = df[df['Title'] <= 'Red']
print(dfnew)

# single condition string isNULL
dfnew = df[df['Country'].isnull()]
print(dfnew)

# single condition string notNULL
dfnew = df[df['Country'].notnull()]
print(dfnew)

# multi condition AND & - notice the brackets 
# each condition MUST be enclosed in seperate brackets
dfnew = df[ (df['Year'] >= 1980) & (df['Year'] <= 1990) ]
print(dfnew)

# multi condition OR | - notice the brackets 
# each condition MUST be enclosed in seperate brackets
dfnew = df[ (df['Year'] <= 1980) | (df['Year'] >= 1990) ]
print(dfnew)

# multi condition AND & + OR | - notice the brackets 
# each condition MUST be enclosed in seperate brackets
dfnew = df[ ( (df['Year'] >= 1980) & (df['Year'] <= 1990) ) | (df['Title'] != 'Red') ]
print(dfnew)

# multi condition NOT ~ 
# each condition MUST be enclosed in seperate brackets
dfnew = df[~( (df['Year'] >= 1980) & (df['Year'] <= 1990) )]
print(dfnew)

# add new column
# assign New Column To Dataframe
df = df.assign(Check='SomeData')
print(df)
#or
df['Check']='SomeData'
print(df)

# assign a new column to df called 'Age' with formula
df['Age']=2020-df['Year']
print(df)

# assign a new column to df called 'Age' with formula
df['Check']="NewData"
print(df)

# also allowed
df['New']=2018-df['Year']
print(df)

# create a new column called Status where the value is based on condition
# if df.age is greater than 30 
df = df.assign(Status="") 
df['Status'] = np.where(df['Age']>=30, 'GoldenOldie', 'RecentHit')
print(df)

# drop columns
# axis : {0 or ‘index’, 1 or ‘columns’}, default 0
# 0 or ‘index’: apply function to each column
# 1 or ‘columns’: apply function to each row
df = df.drop('Company', axis=1)
print(df.info())
print(df)

# drop / delete row based on RowIndex
print(df)
dfnew = df.drop(df.index[2])
print(dfnew)
print(dfnew.head())
print(dfnew[2:3])

# drop / delete row based on conditions
dfnew = df[ df.Title != 'Red' ]
print(dfnew)

# rename column name
# rename the dataframe's column names with a new set of column names 
df = pd.read_csv('./data/Catalog.csv')
df = df.assign(Age=2020-df['Year'])
df['Status'] = np.where(df['Age']>=30, 'GoldenOldie', 'RecentHit')
df.info()

ColNames=['title', 'artist', 'country', 'company', 'price', 'year', 'age', 'label']
df.columns = ColNames
df.info()

# rename specific column name
# rename the dataframe's column names with a new set of column names 
df=df.rename(columns = {'label':'status'})
df.info()

# save daqtaframe to csv
df.to_csv('./data/Sample.csv')
#df.to_csv('./data/Sample.csv', index=False)

# rename column name
# rename the dataframe's column names with a new set of column names 
df = pd.read_csv('./data/Catalog.csv')
df = df.assign(Age=2020-df['Year'])
df['Status'] = np.where(df['Age']>=30, 'GoldenOldie', 'RecentHit')
df.info()

# see the top & bottom rows of the frame
print(df.head(3))
print(df.tail(3))

# check Country Count
print(df.groupby(['Country'])['Company'].count())
print("")
print(df.groupby(['Company'])['Country'].count())
print("")


# check Year Count
print(df.groupby(['Year'])['Title'].count())
print("")

# check Country + Year Count
print(df.groupby(['Country','Year'])['Title'].count())
print("")

# selective update
df['Country'] = np.where(df['Country']=='UK','United Kingdom   ', df['Country'])
df['Country'] = np.where(df['Country']=='USA','United States   ', df['Country'])
df['Country'] = np.where(df['Country']=='EU','Europe Union     ', df['Country'])
print(df)

# check Country Count
print(df.groupby(['Country'])['Title'].count())
print("")
