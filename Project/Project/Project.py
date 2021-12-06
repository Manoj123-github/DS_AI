# -*- coding: utf-8 -*-print(df.head)
"""
Created on Wed Dec  9 07:56:09 2020

@author: Manoj Yadav
"""

import pandas as pd
import numpy as np
import utils
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

 
df = pd.read_csv("C:/Python/Project/iris-m.csv")
print(df.head)

# 1 . What is structure of the dataset .

print(df.columns)
print(df.describe())
print(df.info())

#  02.  
print(df.dtypes)

 # 03.
df['length'] = df['SepalLength'].map(str).apply(len)
print (df)
df['length'] = df['SepalWidth'].map(str).apply(len)
print (df)
df['length'] = df['PetalLength'].map(str).apply(len)
print (df)
df['length'] = df['PetalWidth'].map(str).apply(len)
print (df)
df['length'] = df['species'].apply(len)
print (df)


#04 


    
 # 05. 

# 07.
print('\n*** Columns With Nulls ***')
df.isnull().sum() 
df.isna().sum() 

# 09. 
Significantcol=['SepalLength','SepalWidth','PetalLength','PetalWidth','species']
for g in Significantcol:
    print("\n***"+str(g)+"***")
    print("*Original Count*")
    print("Zero Values: ", (df[g]==0).sum())
    print("Null Values: ",df[g].isnull().sum())
    df[g] = np.where(df[g]==0, df[g].mean(), df[g])
    df[g] = df[g].fillna(df[g].mean())
    print("*Cleaned Count*")
    print("Zero Values: ", (df[g]==0).sum())
    print("Null Values: ",df[g].isnull().sum())


# 10 
print(df['SepalLength'].count())
print(df['SepalWidth'].count())
print(df['PetalLength'].count())
print(df['PetalWidth'].count())
print(df['species'].count())
#print(df.count())

print(df['SepalLength'].mean())
print(df['SepalWidth'].mean())
print(df['PetalLength'].mean())
print(df['PetalWidth'].mean())
#print(df['species'].mean())
# print(df.mean())

print(df['SepalLength'].sum())
print(df['SepalWidth'].sum())
print(df['PetalLength'].sum())
print(df['PetalWidth'].sum())
#print(df.sum())
print(df['SepalLength'].quantile(q=0.25))
print(df['SepalWidth'].quantile(q=0.25))
print(df['PetalLength'].quantile(q=0.25))
print(df['PetalWidth'].quantile(q=0.25))
print(df['SepalLength'].quantile(q=0.5))
print(df['SepalWidth'].quantile(q=0.5))
print(df['PetalLength'].quantile(q=0.5))
print(df['PetalWidth'].quantile(q=0.5))
print(df['SepalLength'].quantile(q=0.75))
print(df['SepalWidth'].quantile(q=0.75))
print(df['PetalLength'].quantile(q=0.75))
print(df['PetalWidth'].quantile(q=0.75))



#11
# print(df['SepalLength'].var())
print(df.var())
#print(df['SepalLength'].range())
df.std()
#12
#Identifying Outliers with Interquartile Range (IQR)
#     The interquartile range (IQR) is a measure of statistical dispersion and 
#     is calculated as the difference between the 75th and 25th percentiles. 
#     It is represented by the formula IQR = Q3 − Q1. 
#    The lines of code below calculate and print the interquartile range for each of the variables in the dataset.
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
print((IQR).count())
print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))

#14
df.hist(
    column=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"],
    figsize=(10, 10))

pylab.suptitle("histrogram", fontsize="xx-large")


#15.
tdf = df.drop(['Id','species'], axis=1)
tdf.head()

plt.figure()
sns.boxplot(tdf)
plt.title(tdf)
plt.ylabel(tdf)
plt.xlabel('Bins')
plt.show()  

# 17.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [df["PetalWidth"], df["PetalLength"]]

ax.scatter(df["PetalWidth"], df["PetalLength"], df["SepalLength"])


ax.set_xlabel('PetalWidthCm')
ax.set_ylabel('PetalLengthCm')
ax.set_zlabel('SepalLengthCm')

plt.tight_layout(pad=0.5)
plt.show()


#18.
Ho = 6.25
# alt hyp
Ha = "at least one of the means is different"
# alpha
al = 0.06

# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
# data
colNames = df['SepalLength']
data = ['SepalLength']
for i, colName in enumerate(colNames):
    print(i)
    print(colName)
    colVals = df[colNames].values.flatten()
    colVals = colVals[~np.isnan(colVals)]
    data.append(colVals)
    print(colVals) 
    print(colVals.mean())          
print("")
rs=f_oneway(data[0],data[1],data[2])
av = rs[0]
pv = rs[1]
print("result",rs)
print("a-stat",av)
print("p-vals",pv)

if pv < al:
    print("Null Hypothesis: Rejected")
    print("Conclusion:",Ha) 
else:
    print("Null Hypothesis: Not Rejected")
    print("Conclusion:",Ho)



#19.
Ho = "mu <= 1.5"
# alt hyp
Ha = "mu > 1.5"
# alpha
al = 0.05
# mu - mean
mu = 1.5
# tail type
tt = 1
# data
PetalWidth = df['PetalWidth'].values
# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
print("mu:", mu)
print(PetalWidth)
print("")
ts, pv = ttest_1samp(PetalWidth, mu)
print("t-stat",ts)
print("p-vals",pv)
t2pv = pv
t1pv = pv*2
print("1t pv",t1pv)
print("2t pv",t2pv)

if tt == 1:
    if t1pv < al:
        print("Null Hypothesis: Rejected")
        print("Conclusion:",Ha)
    else:
        print("Null Hypothesis: Not Rejected")
        print("Conclusion:",Ho)
else:
    if t2pv < al/2:
        print("Null Hypothesis: Rejected")
        print("Conclusion:",Ha)
    else:
        print("Null Hypothesis: Not Rejected")
        print("Conclusion:",Ho) 
