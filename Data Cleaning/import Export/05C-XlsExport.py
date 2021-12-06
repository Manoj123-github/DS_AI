

# import pandas
import pandas as pd

# assign spreadsheet filename: file
file = 'C:\Python\data\import export data\Catalog.csv'

# load spreadsheet xls
df = pd.read_csv(file)

# info
print(df.info())

# summary
print(df.describe())

# create excel write object
# also opens the file
writer = pd.ExcelWriter('C:\Python\data\import export data\nCatalog.xlsx')
# engine to use - you can also set this via the options io.excel.xlsx.writer, io.excel.xls.writer, and io.excel.xlsm.writer.
#writer = pd.ExcelWriter('C:\Python\data\import export data\nCatalog.xlsx',engine="xlsxwriter")

# write data frame
df.to_excel(writer,'Sheet2')

# save & close
writer.save()
writer.close()
