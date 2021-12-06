

# import pandas
import pandas as pd

# assign spreadsheet filename: file
file = 'C:\Python\data\import export data\Catalog.tdl'

# load spreadsheet xls
df = pd.read_csv(file, sep="\t")

# info
print(df.info())

# summary
print(df.describe())

# info
print(df.head())
