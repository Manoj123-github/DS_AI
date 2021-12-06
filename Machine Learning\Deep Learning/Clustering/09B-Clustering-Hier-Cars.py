
# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
import seaborn as sns
import utils

##############################################################
# Read Data 
##############################################################

# read dataset
print("\n*** Read Data ***")
df = pd.read_csv('C:/Users/Manoj Yadav/Desktop/ML/Clustering/cars-data.csv')
print("Done ...")


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

# shape
print("\n*** Shape ***")
print(df.shape)


##############################################################
# Data Transformations
##############################################################

# drop cols
# drop cols which contain identifiers, naminals, descriptions
# change as required
print("\n*** Drop Cols ***")
dfId = df[['Make', 'Model']]                  # store Id in dfID to recreate dataframe later
df = df.drop('Make', axis=1)
df = df.drop('Model', axis=1)
print("Done ...")

# convert string / categoric to numeric
# transformations
# change as required
print("\n*** Transformations ***")
lstLabels = ['Type','Origin','DriveTrain']
for label in lstLabels: 
    df[label] = pd.Categorical(df[label])
    df[label] = df[label].cat.codes
print("Done ...")

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers if required

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required

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

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required

##############################################################
# Visual Data Analytics
##############################################################

# boxplot
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
for colName in colNames:
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# histograms
print("\n*** Histogram Plot ***")
colNames = df.columns.tolist()
print('Histograms')
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# all categoic variables 
# change as required
colNames = ["Origin","DriveTrain","AM"]
print("\n*** Distribution Plot ***")
for colName in colNames:
    plt.figure()
    sns.countplot(df[colName],label="Count")
    plt.title(colName)
    plt.show()


################################
# Hierarchical Clustering
###############################

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

#method=’ward’ 
#uses the Ward variance minimization algorithm. The new entry  is computed as 
#per the formula. This is also known as the incremental algorithm.

# linkage
print("\n*** Linkage Method ***")
from scipy.cluster import hierarchy as hac
vLinkage = hac.linkage(df, 'ward')
print("Done ...")

# make the dendrogram
print("\n*** Plot Dendrogram ***")
print("Looks Cluttered")
plt.figure(figsize=(8,8))
hac.dendrogram(vLinkage, 
               orientation='left')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Linkage (Ward)')
plt.show

# make the dendrogram - large so readable
# make the dendrogram
print("\n*** Plot Dendrogram ***")
print("No Groups")
plt.figure(figsize=(8,80))
hac.dendrogram(vLinkage, 
               leaf_font_size=10.,
               orientation='left')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Linkage (Ward)')
plt.show


# make the dendrogram - truncated
# make the dendrogram
print("\n*** Plot Dendrogram ***")
print("With Groups")
plt.figure(figsize=(8,10))
hac.dendrogram(vLinkage,
               truncate_mode='lastp',   # show only the last p merged clusters
               p=10,                    # show only the last p merged clusters
               leaf_font_size=12.,
               show_contracted=True,    # to get a distribution impression in truncated branches
               orientation='left'       # left to right
               )
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Linkage (Ward)')
plt.show

# create cluster model
print("\n*** Agglomerative Clustering ***")
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')  
# train and group together
lGroups = model.fit_predict(df)
print(lGroups)
# update data frame
df['GroupId'] = lGroups
# recreate data frame
df = pd.concat([dfId, df], axis=1)
print("Done ...")

# check class
# outcome groupby count    
print("\n*** Group Counts ***")
print(df.groupby('GroupId').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(data=df, x='GroupId', label="Count")
plt.title('Group Id')
plt.show()
