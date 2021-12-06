# -*- coding: utf-8 -*-


# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# pandas 
import pandas as pd
# numpy
import numpy as np
# stats
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
from scipy.stats import ttest_rel

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
df = xls.parse('tTest-Dep2SM-2')

# print df
colNames = df.columns.tolist()
for colName in colNames:
    print(df[colName].values.flatten())

##############################################################
# Hypothesis Test
##############################################################

# Problem	Prices of given products in two given cities is same				

#a 0.05
#m-mu - m-nm = 0
#m-mu - m-nm != 0
#TwoTailed
#Test: Dep Two Sample Mean

# null hyp
Ho = "m-mu - m-nm = 0"
# alt hyp
Ha = "m-mu - m-nm != 0"
# alpha
al = 0.05
# mu - mean
#mu = 66
# tail type
tt = 2
# data
d_mu = df['Mumbai'].values
d_nm = df['NMumbai'].values
d_mu = d_mu[~np.isnan(d_mu)]
d_nm = d_nm[~np.isnan(d_nm)]
# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
#print("mu:", mu)
print()
print(d_mu)
print(d_nm)
print("")

ts, pv = ttest_rel(d_mu, d_nm)
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

