
# imports
import pandas as pd
import numpy as np
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
df = pd.read_csv('C:/Users/Manoj Yadav/Desktop/ML/Clustering/online-news.csv')
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
# Data Transformation
##############################################################

# drop cols
# drop cols which contain identifiers, nominals, descriptions
# change as required
print("\n*** Drop Cols ***")
dfId = df[['url',' num_imgs', ' num_videos',' num_keywords',' weekday_is_monday', ' weekday_is_tuesday', ' weekday_is_wednesday',
       ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday',
       ' weekday_is_sunday', ' is_weekend'] ]    # store Id in dfID to recreate dataframe later
df = df.drop(['url',' num_imgs', ' num_videos',' num_keywords',' weekday_is_monday', ' weekday_is_tuesday', ' weekday_is_wednesday',
       ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday',
       ' weekday_is_sunday', ' is_weekend'], axis=1)
print("Done ...")


# columns
print("\n*** Columns ***")
print(df.columns)

# transformations
# change as required
print("\n*** Transformations ***")
print("None ...")

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outlier
colNames = [' n_tokens_title', ' n_tokens_content',
       ' n_unique_tokens', ' n_non_stop_words', ' n_non_stop_unique_tokens',
       ' num_hrefs', ' num_self_hrefs', ' average_token_length',
       ' data_channel_is_lifestyle', ' data_channel_is_entertainment',
       ' data_channel_is_bus', ' data_channel_is_socmed',
       ' data_channel_is_tech', ' data_channel_is_world', ' kw_min_min',
       ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max',
       ' kw_avg_max',' kw_max_avg', ' kw_avg_avg',
       ' self_reference_min_shares', ' self_reference_max_shares',
       ' self_reference_avg_sharess', ' LDA_00',' LDA_01',' global_subjectivity',
       ' global_sentiment_polarity', ' global_rate_positive_words',
       ' global_rate_negative_words',' rate_negative_words', ' avg_positive_polarity',
       ' min_positive_polarity',' avg_negative_polarity',' max_negative_polarity',
       ' title_sentiment_polarity', ' shares']
for colName in colNames:
      colType =  df[colName].dtype  
      df[colName] = utils.HandleOutliers(df[colName])
      if df[colName].isnull().sum() > 0:
          df[colName] = df[colName].astype(np.float64)
      else:
          df[colName] = df[colName].astype(colType)    
    
# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

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
# normalize data
print('\n*** Normalize Data ***')
df = utils.NormalizeData(df)
print('Done ...')

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required


##############################################################
# Visual Data Analytics
##############################################################

# boxplot
print("\n*** Box Plot ***")
plt.figure(figsize=(10,5))
sns.boxplot(data=df)
plt.show()

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



################################
# Prepare Data
################################

# split into data & target
print("\n*** Prepare Data ***")
allCols = df.columns.tolist()
print(allCols)
#allCols.remove(clsVars)
print(allCols)
X = df[allCols].values
#y = df[clsVars].values

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
#print(y.shape)
print(type(X))
#print(type(y))

# head
print("\n*** Prepare Data - Head ***")
print(X[0:4])
#print(y[0:4])

################################
# Knn Clustering
###############################

# imports
from sklearn.cluster import KMeans

# how to decide on the clusters
# within cluster sum of squares errors - wcsse
# elbow method ... iterations should be more than 10
print("\n*** Compute WCSSE ***")
vIters = 20
lWcsse = []
for i in range(1, vIters):
    kmcModel = KMeans(n_clusters=i)
    kmcModel.fit(X)
    lWcsse.append(kmcModel.inertia_)
for vWcsse in lWcsse:
    print(vWcsse)

# plotting the results onto a line graph, allowing us to observe 'The elbow'
print("\n*** Plot WCSSE ***")
plt.figure()
plt.plot(range(1, vIters), lWcsse)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSSE') #within cluster sum of squares error
plt.show()

# programatically
#!pip install kneed
print("\n*** Find Best K ***")
import kneed
kl = kneed.KneeLocator(range(1, vIters), lWcsse, curve="convex", direction="decreasing")
vBestK = kl.elbow
print(vBestK)

# k means cluster model
print("\n*** Model Create & Train ***")
model = KMeans(n_clusters=vBestK, random_state=707)
model.fit(X)

# result
print("\n*** Model Results ***")
print(model.labels_)
df['PredKnn'] = model.labels_

# counts for knn
print("\n*** Counts For Knn ***")
print(df.groupby(df['PredKnn']).size())

# class count plot
print("\n*** Distribution Plot - KNN ***")
plt.figure()
sns.countplot(data=df, x='PredKnn', label="Count")
plt.title('Distribution Plot - KNN')
plt.show()


################################
# Hierarchical Clustering
###############################

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
               p=5,                     # p number of clusters
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
model = AgglomerativeClustering(n_clusters=vBestK, affinity='euclidean', linkage='ward')  
# train and group together
lGroups = model.fit_predict(df)
print(lGroups)
# update data frame
df['PredHeir'] = lGroups
print("Done ...")

# counts for heir
print("\n*** Counts For Heir ***")
print(df.groupby(df['PredHeir']).size())

# class count plot
print("\n*** Distribution Plot - Heir ***")
plt.figure(),
sns.countplot(data=df, x='PredHeir', label="Count")
plt.title('Distribution Plot - Heir')
plt.show()

# counts for knn
print("\n*** Counts For Knn ***")
print(df.groupby(df['PredKnn']).size())

# counts for heir
print("\n*** Counts For Heir ***")
print(df.groupby(df['PredHeir']).size())

