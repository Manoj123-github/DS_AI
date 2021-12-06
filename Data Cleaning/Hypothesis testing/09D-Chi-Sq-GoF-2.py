
# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# pandas 
import pandas as pd
# numpy
import numpy as np
# stats
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
from scipy.stats import chisquare

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
df = xls.parse('Chi-Sq-GoF-2')

# print df
print(df)

##############################################################
# Hypothesis Test
##############################################################

# Title Chi-Square Godness Of Fit	
# Problem: Check if the that the categorical data has the given frequencies.

# alpha	0.05
# Ho	Counts for Sample Given is in 9:3:3:1 ratio
# Ha	Counts for Sample Given is NOT in 9:3:3:1 ratio
# Test	Chi-Square Test Godness Of Fit	

# null hyp
Ho = "Counts for Sample Given is in 9:3:3:1 ratio"
# alt hyp
Ha = "Counts for Sample Given is NOT in 9:3:3:1 ratio"
# alpha
al = 0.05
# data
d_ac = df['ActCount'].values
d_er = df['ExpRatio'].values
d_ec = (d_er/sum(d_er))*sum(d_ac)
# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
print("ActCount:", d_ac)
print("ExpRatio:", d_er)
print("ExpCount:", d_ec)
print("")
cs, pv = chisquare(d_ac, d_ec)
print("cs-stat",cs)
print("p-value",pv)

if pv < al:
    print("Null Hypothesis: Rejected")
    print("Conclusion:",Ha) 
else:
    print("Null Hypothesis: Not Rejected")
    print("Conclusion:",Ho)

