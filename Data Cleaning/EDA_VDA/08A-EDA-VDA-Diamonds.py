

# hides all warnings
import warnings
warnings.filterwarnings('ignore')

# imports
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
# seaborn
import seaborn as sns
# utils
import utils

##############################################################
# Read Data 
##############################################################

# read dataset
df = pd.read_csv('./diamonds-m.csv')


##############################################################
# Exploratory Data Analytics
##############################################################

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

##############################################################
# Transformation / Data Cleaning
##############################################################

# first convert alpha to numeric

# check Cut
# convert alpha to numeric via map
# map can handle errors easily
print("\n*** Cut ***")
colName = 'cut'  
# original data
print("*Original Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())
print("Class Count")
print(df.groupby([colName])['id'].count())
# clean data
print("*Categoric Data*")
print(df[colName].unique())
df[colName] = df[colName].map({"Fair":0, "Good":1, "Very Good":2, "Premium":3, "Ideal":4, "Unknown":-1})
print(df[colName].unique())
# cleaned data
print("*Cleaned Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())
print("Class Count")
print(df.groupby([colName])['id'].count())


# check color
# convert alpha to numeric via cat codes
# cat code can handle nulls but not errors
print("\n*** Color ***")
colName = 'color'  
print("*Original Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())
print("Class Count")
print(df.groupby([colName])['id'].count())
# clean data
print("*Categoric Data*")
print(df[colName].unique())
df[colName] = pd.Categorical(df[colName])
df[colName] = df[colName].cat.codes
print(df[colName].unique())
# cleaned data
print("*Cleaned Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())
print("Class Count")
print(df.groupby([colName])['id'].count())


# check clarity
# convert alpha to numeric via label encoder
# le can't handle nulls & erros ... requires valid data only
print("\n*** Clarity ***")
colName = 'clarity'
print("*Original Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())
print("Class Count")
print(df.groupby([colName])['id'].count())
# clean data
print("*Categoric Data*")
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
print(df[colName].unique())
df[colName] = le.fit_transform(df[colName])
print(df[colName].unique())
# cleaned data
print("*Cleaned Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())
print("Class Count")
print(df.groupby([colName])['id'].count())


# check popularity
# convert alpha to numeric via cat codes
# handle errors & nulls
print("\n*** Popularity ***")
colName = 'popularity'  
print("*Original Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())
print("Class Count")
print(df.groupby([colName])['id'].count())
# clean data
print("*Categoric Data*")
df[colName] = np.where(df[colName] == "NotAvail", None, df[colName])
print(df[colName].unique())
df[colName] = pd.Categorical(df[colName])
df[colName] = df[colName].cat.codes
print(df[colName].unique())
# cleaned data
print("*Cleaned Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())
print("Class Count")
print(df.groupby([colName])['id'].count())



# drop cols
# all ida, names, descriptions to be dropped
# change as required
print("\n*** Drop Cols ***")
df = df.drop('id', axis=1)
print("Done ...")


# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))


# check outlier index
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))


# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers if required

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# check mean
print('\n*** Mean In Columns ***')
print(df.mean())

# handle normalization if required

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required
# check X
print("\n*** Feature x ***")
colName = 'x'
print("*Original Count*")
print("Zero Values: ", (df[colName]==0).sum())
df[colName] = np.where(df[colName]==0, df[colName].mean(), df[colName])
print("*Cleaned Count*")
print("Zero Values: ", (df[colName]==0).sum())


# check y
print("\n*** Feature y ***")
colName = 'y'
print("*Original Count*")
print("Zero Values: ", (df[colName]==0).sum())
df[colName] = np.where(df[colName]==0, df[colName].mean(), df[colName])
print("*Cleaned Count*")
print("Zero Values: ", (df[colName]==0).sum())


# check z
print("\n*** Feature z ***")
colName = 'z'
print("*Original Count*")
print("Zero Values: ", (df[colName]==0).sum())
df[colName] = np.where(df[colName]==0, df[colName].mean(), df[colName])
print("*Cleaned Count*")
print("Zero Values: ", (df[colName]==0).sum())


# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 


# check Carat
# handle nulls if required
print("\n*** Carat ***")
colName = 'carat'
print("*Original Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())
# handle nulls
print("Basic Stats")
print("Mean:   ",df[colName].mean())
print("Median: ",df[colName].median())
print("Mode:   ",df[colName].mode())
# assign lower of mean & median
df[colName] = df[colName].fillna(df[colName].median())
# clean counts
print("*Cleaned Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())

# check Price
print("\n*** Price ***")
colName = 'price'
print("*Original Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())
# handle nulls
print("Basic Stats")
print("Mean:   ",df[colName].mean())
print("Median: ",df[colName].median())
print("Mode:   ",df[colName].mode())
# assign lower of mean & median
df[colName] = df[colName].fillna(df[colName].median())
# clean counts
print("*Cleaned Count*")
print("Null Values: ", df[colName].isnull().sum())
print("Zero Values: ", (df[colName]==0).sum())



# computed depth percentage
# computed depth
df['comp_depth'] = df['z'] / ((df['x']+df['y'])/2)*100
# difference between depth & computed depth
df['depth_diff'] = abs(df['depth']-df['comp_depth'])
# percentage difference between depth & computed depth
df['depth_diff_per'] = (df['depth_diff']/df['depth'])*100
# print
print("\n*** Depth Percentage Difference ***")
# count
print('*Count*')
print(df [ df['depth_diff_per'] >= 5 ]['depth_diff_per'].count())
# values
print('*Values*')
print(df [ df['depth_diff_per'] >= 5 ]['depth_diff_per'])



##############################################################
# Visual Data Analytics
##############################################################

# store dependent variable  
# change as required
depVars = "price"
print(depVars)
print("Done ...")


# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.2f}'.format
print(df.corr())


# check relation with corelation - heatmap
print("\n*** Heat Map ***")
plt.figure(figsize=(8,8))
ax = sns.heatmap(df.corr(), annot=True, cmap="PiYG")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()


# boxplot
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
for colName in colNames:
    if df[colName].dtype == "object": 
        continue
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()


# histograms
# https://www.qimacros.com/histogram-excel/how-to-determine-histogram-bin-interval/
# plot histograms
print('\n*** Histograms ***')
colNames = df.columns.tolist()
for colName in colNames:
    if df[colName].dtype == "object": 
        continue
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()


# scatterplots
# plot Sscatterplot
print('\n*** Scatterplot ***')
colNames = df.columns.tolist()
#colNames.remove(depVars)
print(colName)
for colName in colNames:
    if df[colName].dtype == "object": 
        continue
    colValues = df[colName].values
    plt.figure()
    sns.regplot(data=df, x=depVars, y=colName, color= 'b', scatter_kws={"s": 5})
    plt.title( colName)
    plt.show()


# class count plot
print("\n*** Distribution Plot ***")
# create list of class varibable manually
colNames = ['cut', 'color', 'clarity']
print(colNames)
for colName in colNames:
    print("\n*"+colName+"*")
    print(df.groupby(colName).size())
    print("")
    plt.figure()
    sns.countplot(df[colName],label="Count")
    plt.title(colName)
    plt.show()

