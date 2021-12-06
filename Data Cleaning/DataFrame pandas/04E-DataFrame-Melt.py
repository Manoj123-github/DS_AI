# -*- coding: utf-8 -*-
"""
@filename: DataFrame-Melt&explode.py

@url: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html
@url: https://www.google.com/search?q=python+pandas+melt+images&rlz=1C1CHBF_enIN872IN872&source=lnms&tbm=isch&sa=X&ved=2ahUKEwir3cKHoJHtAhUlzDgGHUDICG4Q_AUoAXoECA8QAw&biw=1139&bih=505#imgrc=gQaKKED16JmiQM
@url: https://medium.com/@durgaswaroop/reshaping-pandas-dataframes-melt-and-unmelt-9f57518c7738
"""

import pandas as pd
import numpy as np


df = pd.DataFrame(data = {
    'Day' : ['MON', 'TUE', 'WED', 'THU', 'FRI'], 
    'Google' : [1129,1132,1134,1152,1152], 
    'Apple' : [191,192,190,190,188] 
})

# print df
print(df)

# info df
print(df.info())

# melt
dfr = df.melt(id_vars=['Day'])

# print dfr
print(dfr)

###########################################################

# create wide dataframe
df = pd.DataFrame(
  {"college":  ["RJC", "SEC", "RJC", "SEC"],
   "student": ["Kavita", "Zeenat", "Minal", "Soumya"],
   "english": [10, 100, 1000, 10000],  # eng grades
   "math":    [20, 200, 2000, 20000],  # math grades
   "physics": [30, 300, 3000, 30000]   # physics grades
  }
)

# print df
print(df)

# info df
print(df.info())

# melt
dfr = df.melt(id_vars=['college','student'])

# print dfr
print(dfr)

###########################################################

# read data from csv and load data in a dataframe
df = pd.read_csv('./data/iip-data.csv')

# print df
print(df)

# info df
print(df.info())

# melt
dfr = pd.melt(df, id_vars =['Item Description']) 

# print dfr
print(dfr)



