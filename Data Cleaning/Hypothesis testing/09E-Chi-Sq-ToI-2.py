

# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# pandas 
import pandas as pd
# numpy
import numpy as np
# stats
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
from scipy.stats import chi2_contingency

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
df = xls.parse('Chi-Sq-ToI-2')

# drop categoric column
df = df.drop('Gender', axis=1)

# print df
print(df)

##############################################################
# Hypothesis Test
##############################################################

# Title Chi-Square Test Of Independence	
# Problem Preference Of Tooth Paste Same In Men & Women Accross Cities

# alpha	0.05
# Ho	Preference Of Tooth Paste Same In Men & Women
# Ha	Preference Of Tooth Paste NOT Same In Men & Women
# Test	Chi-Square Test Test Of Independence

# null hyp
Ho = "Preference Of Tooth Paste Same In Men & Women"
# alt hyp
Ha = "Preference Of Tooth Paste NOT Same In Men & Women"
# alpha
al = 0.05
# actual table
act = df.values.tolist()
# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
print("Actual:", act)
print("")

cs, pv, df, exp = chi2_contingency(act)
print("cs-stat",cs)
print("p-value",pv)

if pv < al:
    print("Null Hypothesis: Rejected")
    print("Conclusion:",Ha) 
else:
    print("Null Hypothesis: Not Rejected")
    print("Conclusion:",Ho)

