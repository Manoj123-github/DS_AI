
# import
import pandas as pd
import numpy as np
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

# reading data file
df = pd.read_csv('./airquality.csv')

# view Col Info
print('\nStructure')
print(df.info())

# view summary
print('\nSummary')
print(df.describe())

# view Data
print('\nHead')
print(df.head())

# check zeros
print('\nColumns With Zero')
print((df==0).sum())

# check nulls
print('\nColumns With Nulls')
df.isnull().sum() 

	
# data imputation - column average for Solar
print('\n************************************************************')
print('Population Mean')
print('************************************************************')

# Option 1: Population Mean - Manual
print('\n*** Option 1: Population Mean - Manual ***')

# check
print('Pre-Clean')
print(df.Solar.isnull().sum())

# mean
vMean = int(df['Solar'].mean())
print('Mean')
print(vMean)

# update
df['Solar'] = np.where(df['Solar'].isnull(), vMean, df['Solar'])
df['Solar'] = df['Solar'].astype(int)

# OR

df['Solar'] = df['Solar'].replace(np.nan, vMean)
df['Solar'] = df['Solar'].astype(int)

# recheck
print('PostClean')
print(df.Solar.isnull().sum())

# mean
vMean = int(df['Solar'].mean())
print('Mean')
print(vMean)


# Option 2:  Population Mean - FillNA-DF  
print('\n*** Option 2: Population Mean - FillNA-DF ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check nulls
print('Pre-Clean')
print('Columns With Nulls')
print(df.isnull().sum()) 

print('Col Means')
print(df.mean().astype(int))

# claen
df = df.fillna(df.mean().astype(int))

# check nulls
print('Post-Clean')
print('Columns With Nulls')
df.isnull().sum() 

print('Col Means')
print(df.mean().astype(int))


# Option 3: Population Mean - FillNA-Col  
print('\n*** Option 3: Population Mean - FillNA-Col ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check nulls
print('Pre-Clean')
print('Columns With Nulls')
print(df['Solar'].isnull().sum())
print('Mean Solar')
vMean = int(df['Solar'].mean())
print(vMean)

# claen
df['Solar'] = df['Solar'].fillna(df['Solar'].mean())
df['Solar'] = df['Solar'].astype(int)

# check nulls
print('Post-Clean')
print('Columns With Nulls')
print(df['Solar'].isnull().sum())
print('Mean Solar')
vMean = int(df['Solar'].mean())
print(vMean)


# Option 4: Population Mean - Imputer-Col
print('\n*** Option 4: Population Mean - Imputer-Col ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check
print('Pre-Clean')
print(df['Solar'].isnull().sum())
print('Mean Solar')
vMean = int(df['Solar'].mean())
print(vMean)

# impute
imp=SimpleImputer(missing_values=np.nan, strategy='mean')
#print(imp)
df['Solar']=imp.fit_transform(df[['Solar']])
df['Solar']=df['Solar'].astype(int)

# recheck
print('PostClean')
print(df['Solar'].isnull().sum())
print('Mean Solar')
vMean = int(df['Solar'].mean())
print(vMean)


# data imputation - group mean for Ozone
print('\n************************************************************')
print('Group Mean')
print('************************************************************')

# Option 1: Group Mean - Manual
print('\n*** Option 1: Group Mean - Manual ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check
print('Pre-Clean')
print(df.Ozone.isnull().sum())
# creates data frame
print('Group Mean')
gdf = df.groupby(['Month'])[['Ozone']].mean().astype(int)
print(gdf)
print(type(gdf))

# iterate
for i in range(0,len(gdf)):
    print(i)    
    # get month
    vMonth = gdf.index[i]
    print(vMonth)
    #get mean
    vMeans = gdf['Ozone'][vMonth]
    print(vMeans)
    # update
    df['Ozone'] = np.where(((df['Ozone'].isnull()) & (df['Month']==vMonth)), 
                                            vMeans, df['Ozone'])
df['Ozone'] = df['Ozone'].astype(int)


# recheck
print('PostClean')
print(df.Ozone.isnull().sum())
print('')
# mean
# creates data frame
print('Group Mean')
cdf = df.groupby(['Month'])[['Ozone']].mean().astype(int)
print(gdf)
print(cdf)

# Option 1: Group Mean - Manual
print('\n*** Option 2: Group Mean - GroupBy Lambda ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check
print('Pre-Clean')
print(df.Ozone.isnull().sum())
# creates data frame
print('Group Mean')
gdf = df.groupby(['Month'])[['Ozone']].mean().astype(int)
print(gdf)
print(type(gdf))

# update
df['Ozone'] = df.groupby('Month')['Ozone'].apply(lambda x:x.fillna(x.mean()))
df['Ozone'] = df['Ozone'].astype(int)

# recheck
print('PostClean')
print(df.Ozone.isnull().sum())
# mean
# creates data frame
print('Group Mean')
cdf = df.groupby(['Month'])[['Ozone']].mean().astype(int)
#print(gdf)
print(cdf)


# data imputation - average for immediate up / down points - Temp
print('************************************************************')
print('Interpolated Mean')
print('************************************************************')

# Option 1: Interpolated Mean - Manual
print('\n*** Option 1: Interpolated Mean - Manual ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check
print('Pre-Clean')
print(df.Temp.isnull().sum())

# handling temp
i = 0
for i in df.index:
    # check if row,temp = null
    if (pd.isnull(df.iloc[i,3])):
        # get previous row value of temp
        vLower = None
        if i > 0:
           vLower = df.iloc[i-1,3] 
        # get next row value of temp
        vUpper = None
        if i < len(df):
           vUpper = df.iloc[i+1,3] 
        vList = [vLower, vUpper]
        vMean = np.mean(vList).astype(int)
        df.iloc[i,3] = vMean
df['Temp'] = df['Temp'].astype(int)

# recheck
print('PostClean')
print(df.Temp.isnull().sum())
print('')


# Option 2: Interpolated  Mean - Interpolate-Col
print('\n*** Option 2: Interpolated  Mean - Interpolate-Col  ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check
print('Pre-Clean')
print(df.Temp.isnull().sum())

# update
df['Temp'] = df['Temp'].interpolate(method ='linear', limit_direction ='forward') 
df['Temp'] = df['Temp'].astype(int)

# recheck
print('PostClean')
print(df.Temp.isnull().sum())
print('')


# Option 3: Interpolated  Mean - Interpolate-DF
print('\n*** Option 3: Interpolated  Mean - Interpolate-DF  ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check
print('Pre-Clean')
print(df.isnull().sum())

# update
df = df.interpolate(method ='linear', limit_direction ='forward') 
df = df.astype(int)

# recheck
print('PostClean')
print(df.isnull().sum())
print('')


# data imputation - populate NAs with dummy values -1
print('************************************************************')
print('Populate NAs With Dummy Values')
print('************************************************************')

# Option 1: Populate All NAs In df With Dummy Value <-1> 
print('\n*** Option 1:  Populate All NAs In df With Dummy Value <-1> ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check NAs
print("\n*** Columns With Nulls")
print(df.isnull().sum()) 

# drop NaN
df = df.fillna(-1)

# check NAs
print("\n*** Columns With Nulls")
print(df.isnull().sum()) 


# Option 1: Populate All NAs In Specific Column With Dummy Value <-1>
print('\n*** Option 2: Populate All NAs In Specific Column With Dummy Value <-1> ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check NAs
print("\n*** Columns With Nulls")
print(df.isnull().sum()) 

# drop NaN
df['Ozone'] = df['Ozone'].fillna(-1)

# check NAs
print("\n*** Columns With Nulls")
print(df.isnull().sum()) 


# data imputation - drop rows
print('************************************************************')
print('Drop Rows With NaNs')
print('************************************************************')

# Option 1: Drop Rows Any Cols with NaNs
print('\n*** Option 1: Drop Rows Any Cols with NaNs  ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check NAs
print("\n*** Row Count ***")
print(len(df.index)) 
print("\n*** Columns With Nulls ***")
print(df.isnull().sum()) 

# drop NaN
df = df.dropna()

# check NAs
print("\n*** Row Count ***")
print(len(df.index)) 
print("\n*** Columns With Nulls ***")
print(df.isnull().sum()) 

# Option 1: Drop Rows With Specific Cols with NaNs
print('\n*** Option 2: Drop Rows With Specific Cols with NaNs  ***')

# reading data file
df = pd.read_csv('./airquality.csv')

# check NAs
print("\n*** Row Count ***")
print(len(df.index)) 
print("\n*** Columns With Nulls ***")
print(df.isnull().sum()) 

# drop NaN
df = df['Ozone'].dropna()

# check NAs
print("\n*** Row Count ***")
print(len(df.index)) 
print("\n*** Columns With Nulls ***")
print(df.isnull().sum()) 

0