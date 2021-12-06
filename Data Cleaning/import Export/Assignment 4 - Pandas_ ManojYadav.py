# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:02:24 2020

@author: Manoj Yadav
"""

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Python/assignment/A04A-Pandas-Marks.csv') 
print(df.head())

#01.  info - The info cammand is used to print a concise summary of a DataFrame
print(df.info())

# 02.  describe- describe() is used to view some basic statistical details like percentile,
#                mean, std etc. of a data frame or a series of numeric values.

 desc = df['total'].describe()
 print(desc)
 
 #03.  Create index for the dataframe using column name

 df= df.set_index('name')
 print(df.head())
 #04.  Remove index and bring back name as normal column dataframe.
  df = df.reset_index() 
  print(df.head())

# 05. For all numeric columns convert all NaNs to 0
df.replace(np.NaN,0)


#06.  Create a new dataframe which contains all records of students who have "qualified"

dfn = df[df['qualify'] == "yes" ]
print(dfn.head())

#07 Create a new dataframe which contains all records of the student 
#  whose attempts is more than 1 or total is more than 40.
dfn = df[(df['attempts'] >= 2 ) | (df['total'] > 40 )]
print(dfn.head())

#08 Create a new dataframe which contains all records names starting with K or names starting with J 
search = "K" ,"J"#
dfn = df["name"].str.startswith(search)
df[dfn]

#09.  Create a new dataframe which contains 
#   either
#  all records of “Male” gender whose total marks is more than 30.
#  OR
#  all records of “Female” gender whose total marks is more than 35.  
dfn = df[((df['gender'] == "Male" ) & (df['total'] > 30 )) | ((df['gender'] == "Female" ) &(df['total'] > 35 ))]
print(dfn.head())


#10. Create a new dataframe which contains all records of “Male”  whose subject1 is less than 13 or subject2 is less than 10.
 dfn = df[((df['gender'] == "Male" ) &(df['subject1'] < 13 )) | ((df['gender'] == "Male" ) &(df['subject2'] < 10 ))]
print(dfn.head())

#11 Create a dataframe containing records James & Jonas.
dfn = df[(df['name'] == "James" ) | (df['name'] == "Jones")]
print(dfn.head())

#12. Add a new column which gives approximate year in which each person was born. (use formula current year - age)


df['birth'] = 2020 - df['age']
print(df.head())

# 13. Create a dataframe and rename all the columns to fname, sex, years, sub1, sub2, sub3, tot, atmpts, qual, yob
df=df.rename(columns = {'name':'fname','gender':'sex','age':'years','subject1':'sub1','subject2':'sub2','subject3':'sub3','total':'tot','attempts':'atmpts','qualify':'qual','birth':'yob'})
                    
df.info()

#14  In the above table, change the order of the columns to fname, sex, years, atmpts, qual, sub1, sub2, sub3, tot, yob 
#df.columns=['fname','sex','years','sub1','sub2','sub3','tot','atmpts','qual','yob']
df=df[['fname','sex','years','atmpts','qual','sub1','sub2','sub3','tot','yob']]
print(df.head())


# 15. Add a new column “sal" (salutation, for sex “Male” / “Female”  salutaion to be “Mr” / “Ms” repectively.
df['sal']=''

print(df.head())
s=df['sex']
c=0
t=4
for i in s:
    if i=='Female':
        df.iloc[c:t,10:]='Ms'
    else:
        df.iloc[c:t,10:]='Mr'
    c=c+1
    t=t+1
print(df.head())
df=df[['sal','fname','sex','years','atmpts','qual','sub1','sub2','sub3','tot','yob']]
print(df.head())


#16 print average ‘total’ of all students who have 'qualified'
df=df[df['qual']=='yes']
a=df['tot']
avg=a.sum()/len(a)
print(avg)


#17 Print median ‘age’ of all persons whose ‘salutation’ equals “Ms”

#s=df['sal']
#or i in s:
#    if i== 'Ms':
 #      df.loc[:,'years'].median()


#18 Create a new dataframe which contains all records but only columns: fname, sex, sub1, sub2, sub3.
df=df[df.columns[0:4]]
print(df.head())


# 19 Create a dataframe in raw format for table generated in point 18 above. Preserve column fname & sex. (hint: use melt)
df = df.melt(id_vars=['fname','sex'])
print(df.head())


#20 Save datafram to "new_marks.csv" without the index column

df.to_csv('r','C:/Python/assignment/new_marks.csv', index = False,header=True)
print(df.head())