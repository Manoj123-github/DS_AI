

# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# pandas 
import pandas as pd
# numpy
import numpy as np
# stats
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
from scipy.stats import ttest_ind

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
df = xls.parse('tTest-Ind2SM-2')

# print df
colNames = df.columns.tolist()
for colName in colNames:
    print(df[colName].values.flatten())

##############################################################
# Hypothesis Test
##############################################################

# Problem: Are GPA scores of male students same as female students					

#a	0.05
#Ho	m-fs - m-ms = 0
#Ha	m-fs - m-ms != 0
#Tail Type	Two
#Test: Ind Two Sample Mean

# null hyp
Ho = "m-fs - m-ms = 0"
# alt hyp
Ha = "m-fs - m-ms !- 0"
# alpha
al = 0.05
# mu - mean
#mu = 66
# tail type
tt = 2
# data
d_fs = df['Female'].values
d_ms = df['Male'].values
d_fs = d_fs[~np.isnan(d_fs)]
d_ms = d_ms[~np.isnan(d_ms)]

# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
#print("mu:", mu)
print(d_fs)
print(d_ms)
print("")
ts, pv = ttest_ind(d_fs, d_ms)
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
