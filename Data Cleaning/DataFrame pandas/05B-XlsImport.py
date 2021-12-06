

# import pandas
import pandas as pd

# assign spreadsheet filename: file
file = './data/leads.xlsx'

# load spreadsheet xls
xls = pd.ExcelFile(file)

# print sheet names
print(xls.sheet_names)

# load a sheet into a dataFrame by name
dfTel = xls.parse('Telephone')

print(dfTel)

# print the head of the dataFrame df1
print(dfTel.head())

# load a sheet into a dataFrame by name
dfChat = xls.parse('WebChat')

# print the head of the dataFrame df1
print(dfChat.head())

# load a sheet into a dataFrame by name
dfForm = xls.parse('WebForm')

# print the head of the dataFrame df1
print(dfForm.head())

# print sheet names
print(xls.sheet_names)

# load a sheet into a dataFrame by index
df0 = xls.parse(0)
# print the head of the DataFrame df0
print(df0.head())

# load a sheet into a dataFrame by index
df1 = xls.parse(1)
# print the head of the DataFrame df1
print(df1.head())

# load a sheet into a dataFrame by index
df2 = xls.parse(2)
# print the head of the DataFrame df2
print(df2.head())

# load a sheet into a dataFrame by index
df3 = xls.parse(3)

# print sheet names
print(xls.sheet_names)

# create an array of dataframes (actually dict)
dfn={}
for i in range(0,len(xls.sheet_names)):
    print(i)
    print(xls.sheet_names[i])
    dfn[i] = xls.parse(i)
print(dfn)