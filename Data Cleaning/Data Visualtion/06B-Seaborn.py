

# pandas for managing datasets
import pandas as pd
# matplotlib for additional customization
from matplotlib import pyplot as plt
#%matplotlib inline
# seaborn for plotting and styling
import seaborn as sns

# read dataset
df = pd.read_csv('C:\Python\data\Data visulation\pokemon.csv', index_col=0)

# display first 5 observations
df.head()

# scatter plot
plt.figure()
sns.lmplot(data=df, x='Attack', y='Defense')
plt.show()

# now extras
plt.figure()
sns.lmplot(x='Attack', y='Defense', data=df,
           fit_reg=False, # No regression line
           hue='Stage')
plt.show()

# plot using Seaborn
plt.figure()
sns.lmplot(x='Attack', y='Defense', data=df,
           fit_reg=False, 
           hue='Stage')
# tweak using matplotlib
plt.ylim(-25, None)
plt.xlim(-25, None)
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
# good practice
plt.show()

# boxplot
plt.figure()
sns.boxplot(data=df)
plt.show()

# temp df with select cols
tdf = df.drop(['Total', 'Stage', 'Legendary'], axis=1)
tdf.head()
# new boxplot using stats_df
plt.figure()
sns.boxplot(data=tdf)
plt.show()

# boxplot of single col
plt.figure()
sns.boxplot(y=df['Total'])
plt.show()

# histogram
plt.figure()
sns.distplot(df.Attack, bins=7, color='r')
sns.distplot(df.Attack, kde=False, bins=7, color='r')
sns.distplot(df.Attack, hist=False, kde=True, bins=7, color='r', vertical=False)
plt.show()

# frequency plot 
plt.figure()
sns.countplot(x='Type1', data=df) 
# Rotate x-labels
plt.xticks(rotation=60)
plt.show()

# basic colors
# https://matplotlib.org/2.0.2/api/colors_api.html
# named colors
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html

# create palette
type_palette = ['#78C850',  # Grass
                '#F08030',  # Fire
                '#6890F0',  # Water
                '#A8B820',  # Bug
                '#A8A878',  # Normal
                '#A040A0',  # Poison
                '#F8D030',  # Electric
                '#E0C068',  # Ground
                '#EE99AC',  # Fairy
                '#C03028',  # Fighting
                '#F85888',  # Psychic
                '#B8A038',  # Rock
                '#705898',  # Ghost
                '#98D8D8',  # Ice
                '#7038F8'  # Dragon
                ]

# frequency plot 
plt.figure()
sns.countplot(x='Type1', data=df, palette=type_palette)
# Rotate x-labels
plt.xticks(rotation=0)
#plt.legend()
plt.show()

# heatmap
# temp df with select cols
tdf = df.drop(['Total', 'Stage', 'Legendary'], axis=1)
tdf.head()
# 
corr = tdf.corr()
print(corr)
# calculate correlations
plt.figure()
# plot heatmap
sns.heatmap(corr)
plt.show()

# read dataset
nifty = pd.read_csv('C:/Python/data/Data visulation/nifty-data.csv')
print(nifty.head())

# verticle bar
dfn = nifty[0:9]
plt.figure()
sns.barplot(x="Symbol", y="OpenPrice", data=dfn,
            label="OpenPrice", color="b")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# horizontal bar
# check this & revert
plt.figure()
dfn = nifty[0:9]
sns.barplot(y="Symbol", x="OpenPrice", data=dfn,
            orient="h", label="OpenPrice", color="b")
plt.xticks(rotation=45)
plt.show()

# group bar
dfn = nifty[0:9]

dfn = dfn[['Symbol','OpenPrice',"HighPrice", "LowPrice",'ClosePrice']]

dfn = pd.melt(dfn, id_vars=["Symbol"])
dfn.columns = ["Symbol","PType","Price"]
dfn.head()

# bar plot
plt.figure()
sns.barplot(x="Symbol", y="Price", data=dfn)
plt.xticks(rotation=45)
plt.show()

# group bar
# check & revert
dfn = nifty[0:9]
dfn = dfn[['Symbol','OpenPrice',"HighPrice", "LowPrice",'ClosePrice']]
dfn = pd.melt(dfn, id_vars=["Symbol"])
dfn.columns = ["Symbol","PType","Price"]
plt.figure()
sns.barplot(data=dfn, x="Symbol", y="Price", hue="PType")
plt.xticks(rotation=45)
plt.show()

# line chart
dfn = nifty[nifty['Symbol'] == 'ACC']
dfn.head()
plt.figure()
sns.pointplot(x="RecDate", y="ClosePrice", data=dfn, 
              color="b", scale=0.5)
plt.xticks(rotation=60)
ax = plt.axes()
#print(ax.get_xticks())
ax.set_xticks(ax.get_xticks()[::3])
ax.set_xticklabels(dfn['RecDate'][::3])
plt.show()

# multi line
dfn = nifty[ (nifty['Symbol'] == 'TCS') | (nifty['Symbol'] == 'INFY') | (nifty['Symbol'] == 'WIPRO') ]
print(dfn[['RecDate','Symbol','ClosePrice']])
dfn.head()
plt.figure()
sns.pointplot(x="RecDate", y="ClosePrice", hue="Symbol", 
                      data=dfn, scale=0.5)
plt.xticks(rotation=60)
ax = plt.axes()
ax.set_xticks(ax.get_xticks()[::3])
ax.set_xticklabels(dfn['RecDate'][::9])
plt.show()

# read dataset
nifty = pd.read_csv('C:/Python/data/Data visulation/nifty-data.csv')
print(nifty.head())

