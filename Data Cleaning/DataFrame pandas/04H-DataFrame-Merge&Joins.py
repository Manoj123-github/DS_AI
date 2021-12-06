

# topics to cover
# row bind
# col bind
# col bind with key
# joins (all)

# import modules
import pandas as pd

# The concat function (in the main pandas namespace) does all of the heavy work 
# of performing concatenation operations along an axis while performing optional 
# set logic (union or intersection) of the indexes (if any) on the other axes. 
# lets start with a simple example:

# default concat
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                    index=[0, 1, 2, 3])
print(df1)

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                     index=[4, 5, 6, 7])
print(df2)

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                    index=range(8,12))
print(df3)

# concat
frames = [df1, df2, df3]
result = pd.concat(frames)
print(result)

# with keys clause
# adds keys to dataframe in addition to row-id
# generally not to be used
result = pd.concat(frames, keys=['x', 'y', 'z'])
print(result)

# define new df
df4 = pd.DataFrame({'E': ['B2', 'B3', 'B6', 'B7'],
                    'F': ['D2', 'D3', 'D6', 'D7'],
                    'G': ['F2', 'F3', 'F6', 'F7']},
                    index=[2, 3, 6, 7])
print(df4)

# concat with axis=0 (default) ie row wise
result = pd.concat([df1, df4])
print(result)

# concat with axis=0 (default) ie row wise
result = pd.concat([df1, df4], axis=0)
print(result)

# concat with axis=1 ie col wise
print(df1)
print(df4)
result = pd.concat([df1, df4], axis=1)
print(result)

# define new df
df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                    'C': ['D2', 'D3', 'D6', 'D7'],
                    'D': ['F2', 'F3', 'F6', 'F7']},
                    index=[2, 3, 6, 7])


# concat with axis=0 (default) ie row wise & join=outer (default) i.e. Set Union 
print(df1)
print(df4)
result = pd.concat([df1, df4], axis=0, join='outer') 
print(result)

# concat with axis=0 (default) ie row wise & join=inner i.e. Set Intersection
print(df1)
print(df4)
result = pd.concat([df1, df4], axis=0, join='inner') 
print(result)

# concat with axis=1 ie col wise & join=outer (default) i.e. Set Union 
result = pd.concat([df1, df4], axis=1, join='outer') 
print(result)

# concat with axis=1 ie row wise & join=inner i.e. Set Intersection
result = pd.concat([df1, df4], axis=0, join='outer') 
print(result)
result = pd.concat([df1, df4], axis=1, join='inner') 
print(result)

# Joining columns based on key-col
# vlook-up equivalent merge cols of df based on key-col
# Merge    SQL Join        Name	Description
# left	    LEFT OUTER JOIN	Use keys from left frame only
# right	RIGHT OUTER JOIN	Use keys from right frame only
# outer	FULL OUTER JOIN	Use union of keys from both frames
# inner	INNER JOIN	       Use intersection of keys from both frames

# define df
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K4'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

#
print(left)
print(right)

# merge cols based on key, inner join
result = pd.merge(left, right, on='key', how='inner')
print(result)

# merge cols based on key, inner left
result = pd.merge(left, right, on='key', how='left')
print(result)

# merge cols based on key, right join
result = pd.merge(left, right, on='key', how='right')
print(result)

# merge cols based on key, outer join
result = pd.merge(left, right, on='key', how='outer')
print(result)
