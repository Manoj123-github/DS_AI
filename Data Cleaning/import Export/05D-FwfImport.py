
# import pandas
import pandas as pd

# define column position
colSpecs=[[0,19], [20,39], [40,59], [60,79], [80,89], [90,94]]

# define column names
colNames=["Album","Artist","Country","Company","Price","Year"]

# read fixed width file
df = pd.read_fwf("C:\Python\data\import export data\Catalog.fwf", 
                 colspecs=colSpecs, 
                 header=None, 
                 names=colNames)

# info 
print(df) 

# summary
print(df.describe()) 

# head
print(df.head()) 