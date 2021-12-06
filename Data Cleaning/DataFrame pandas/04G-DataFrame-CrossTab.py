
# imports
# pandas 
import pandas as pd


# read dataset
df = pd.read_csv('./data/auto-cars.csv', na_values="?")

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

# counts
print("\n*** Counts ***")
print(df.groupby('make').size())
 
# counts
print("\n*** All Counts ***")
print(df.groupby('make').count())

# get unique 
print("\n*** Unique Make ***")
lsMake = (df.make.unique())
print(lsMake)

# head
print("\n*** Head Again ***")
print(df.head())

# head
print("\n*** Print As Rows & Cols - Three Ways***")
# group by
print(df.groupby(['make', 'body_style'])['body_style'].count().unstack().fillna(0))
# cross tab
print(pd.crosstab(df.make, df.body_style))

# cross tab - col & row totals
print(pd.crosstab(df.make, df.num_doors, margins=True))
print(pd.crosstab(df.make, df.num_doors, margins=True, margins_name="Total"))

# cross tab - aggregation
print(pd.crosstab(df.make, df.body_style, values=df.curb_weight, aggfunc='mean'))

# cross tab - rounding
print(pd.crosstab(df.make, df.body_style, values=df.curb_weight, aggfunc='mean').round(0))
print(pd.crosstab(df.make, df.body_style, values=df.curb_weight, aggfunc='mean').round(2))
print(pd.crosstab(df.make, df.body_style, values=df.curb_weight, aggfunc='mean').round(-2))

# cross tab - rounding
print(pd.crosstab(df.make, df.body_style, values=df.curb_weight, aggfunc='sum'))

# percent of total overall
pd.crosstab(df.make, df.body_style, normalize=True)
# percent of total col-wise
pd.crosstab(df.make, df.body_style, normalize='columns')
# percent of total row-wise
pd.crosstab(df.make, df.body_style, normalize='index')

# grouping in col
pd.crosstab(df.make, [df.body_style, df.drive_wheels])
dfg = pd.crosstab(df.make, [df.body_style, df.drive_wheels])

# grouping in row
pd.crosstab([df.make, df.num_doors], [df.body_style])
dfg = pd.crosstab([df.make, df.num_doors], [df.body_style])

pd.crosstab([df.make, df.num_doors], [df.body_style, df.drive_wheels],
            rownames=['Auto Manufacturer', "Doors"],
            colnames=['Body Style', "Drive Type"],
            dropna=False)
dfg = pd.crosstab([df.make, df.num_doors], [df.body_style, df.drive_wheels],
            rownames=['Auto Manufacturer', "Doors"],
            colnames=['Body Style', "Drive Type"],
            dropna=False)
