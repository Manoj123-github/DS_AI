

# import pandas
import pandas as pd
import numpy as np

# assign spreadsheet filename: file
file = 'C:\Python\data\import export data\patient-data.xlsx'

# create xls object
xls = pd.ExcelFile(file)

# print sheet names
print(xls.sheet_names)

# load a sheet into a dataFrame by name
df = xls.parse('Original')

# print the head of the dataFrame df1
print(df.head())

# assign "" to cols Name & State
df['Salutation'] = np.where(df['Gender'].str.strip()=="Male","Mr","Ms")

# create excel write object (also opens the file)
writer = pd.ExcelWriter('C:\Python\data\import export data\patient-data-mod.xlsx')

# write data frame
# usages : dataframe.to_excel(writer-object,"sheetname", index=False)
dfo = xls.parse('Original')
dfo.to_excel(writer,'Original', index=False)
df.to_excel(writer,'Modified', index=False)

# save & close
writer.save()
writer.close()
