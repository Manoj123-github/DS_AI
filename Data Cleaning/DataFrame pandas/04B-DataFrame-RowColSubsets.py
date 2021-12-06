

# import the pandas module
import pandas as pd

# create an sample dataframe 
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
            'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
            'deaths': [523, 52, 25, 616, 43, 234, 523, 62, 62, 73, 37, 35],
            'battles': [5, 42, 2, 2, 4, 7, 8, 3, 4, 7, 8, 9],
            'size': [1045, 957, 1099, 1400, 1592, 1006, 987, 849, 973, 1005, 1099, 1523],
            'origin': ['Arizona', 'California', 'Texas', 'Florida', 'Arizona', 'Arizona', 'Alaska', 'Washington', 'Oregon', 'Wyoming', 'Louisana', 'Georgia']}

# create dataframe
df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'deaths', 'battles', 'size', 'origin'])
print(df) 

# create index
df = df.set_index('origin')
print(df) 


# select a column
dfn = df['size']
print(dfn)
print(type(dfn))

dfp = df[['size']]
print(dfp)
print(type(dfp))

# select two or more columns
dfp = df[['size', 'battles']]
print(dfp)
print(type(dfp))
dfp = df[['size', 'battles','deaths']]
print(dfp)
print(type(dfp))

# show index item 
# select every index item up to 3
print(df)
print(df.index[0:7])
print(df.index[4])
print(df.index[4:])
print(df.index[4:-1])
print(df.index[0:20])
print(df.index[20])


# remove index
df = df.reset_index() 
#or
#df.reset_index(inplace = True) 
print(df)
print(type(df))


# iloc
# iloc performs integer-based access to the index, purely by position: 
# that is, if you think of the of the DataFrame as a list of values or rows
# iloc does normal 0-based list access.

# select rows by row number
# select every row up to 1
print(df)
dft = df.iloc[:2]
print(dft)
print(type(dft))

# select rows by row number
# select every row up to 1
print(df)
dft = df.iloc[0:2]
print(dft)
print(type(dft))

# select the second df.iloc[1:2]
print(df)
dft = df.iloc[1:2]
print(dft)
print(type(dft))

# select every row after the third row
print(df)
dft = df.iloc[2:]
print(dft)
print(type(dft))

print(df)
dft = df.iloc[2:-1]
print(dft)
print(type(dft))


# select columns by column number
# select the first 2 columns
print(df)
dft = df.iloc[:,:]
print(dft)
print(type(dft))

# very important - why?
# result is not dataframe but a basic object like int, float, string, bool 
print(df)
tmp = df.iloc[2,2]
print(tmp)
print(type(tmp))

# very important - why?
print(df)
tmp = df.iloc[2:3,2:3]
print(tmp)
print(type(tmp))

# select by conditionals (boolean)
# select rows where df.deaths is greater than 100
print(df)
dfp = df[   df['deaths'] > 100  ]
print(dfp)
print(type(dfp))

print(df)
dfp = df[df['deaths'] > 100].iloc[1:3,]
print(dfp)
print(type(dfp))

