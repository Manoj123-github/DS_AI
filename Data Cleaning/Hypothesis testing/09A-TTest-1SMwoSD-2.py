
# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# pandas 
import pandas as pd
# numpy
import numpy as np
# stats
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
from scipy.stats import ttest_1samp

##############################################################
# Read Data 
##############################################################

# assign spreadsheet filename: file
file = './data/ttest-data.xlsx'

# load spreadsheet xls
xls = pd.ExcelFile(file)

# print sheet names
print(xls.sheet_names)

# load a sheet into a dataFrame by name
df = xls.parse('tTest-1SMwoSD-2')

# print df
print(df.values.flatten())

##############################################################
# Hypothesis Test
##############################################################

# Problem: Check if the population mean age is  equal to 66

#Ho:	m = 66
#Ha:	m != 66
#Tail: Two
#Test: One Sample Mean without std

# null hyp
Ho = "mu = 66"
# alt hyp
Ha = "mu != 66"
# alpha
al = 0.05
# mu - mean
mu = 66
# tail type
tt = 2
# data
ages = df['Age'].values
# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
print("mu:", mu)
print(ages)
print("")
ts, pv = ttest_1samp(ages, mu)
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
