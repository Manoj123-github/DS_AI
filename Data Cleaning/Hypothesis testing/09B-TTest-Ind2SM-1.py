

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
df = xls.parse('tTest-Ind2SM-1')

# print df
colNames = df.columns.tolist()
for colName in colNames:
    print(df[colName].values.flatten())

##############################################################
# Hypothesis Test
##############################################################

# Problem: Do full time students spend more time studying than part time students					

#a	0.05
#Ho	m-ft - m-pt >= 0
#Ha	m-ft - m-pt < 0
#Tail Type	One
#Test: Ind Two Sample Mean

# null hyp
Ho = "m-ft - m-pt >= 0"
# alt hyp
Ha = "m-ft - m-pt < 0"
# alpha
al = 0.05
# mu - mean
#mu = 66
# tail type
tt = 1
# data
d_ft = df['FullTime'].values
d_pt = df['PartTime'].values
d_ft = d_ft[~np.isnan(d_ft)]
d_pt = d_pt[~np.isnan(d_pt)]

# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
#print("mu:", mu)
print(d_ft)
print(d_pt)
print("")
ts, pv = ttest_ind(d_ft, d_pt)
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
