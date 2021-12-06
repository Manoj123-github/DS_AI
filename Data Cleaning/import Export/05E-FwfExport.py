

# import pandas
import pandas as pd
import numpy as np

# assign spreadsheet filename: file
file = 'C:\Python\data\import export data\Catalog.csv'

# load spreadsheet xls
df = pd.read_csv(file)

# info
print(df.info())

# summary
print(df.describe())

# info
print(df.head())

# prepare for write fwf
cFrmt = '%5s%-20s%-20s%-20s%-20s%10.1f%4d'

# write fwf    
np.savetxt(r'C:\Python\data\import export data\nCatalog.fwf', df.to_records(), fmt=cFrmt, delimiter='')
