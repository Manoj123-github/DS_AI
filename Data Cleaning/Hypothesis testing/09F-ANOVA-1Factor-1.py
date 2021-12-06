

# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# pandas 
import pandas as pd
# numpy
import numpy as np
# stats
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
from scipy.stats import f_oneway

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
df = xls.parse('Anova-1Factor-1')

# print df
colNames = df.columns.tolist()
for colName in colNames:
    print(df[colName].values.flatten())

##############################################################
# ANOVA Test
##############################################################

# Title Chi-Square Godness Of Fit	
# Problem	Mean of the marks obtained students (not same) of three subjects are same / similar


# alpha	0.05
# Ho	μ1 = μ2 = μ3
# Ha	at least one of the means is different
# Test	Annova: Single Factor

# null hyp
Ho = "μ1 = μ2 = μ3"
# alt hyp
Ha = "at least one of the means is different"
# alpha
al = 0.05

# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
# data
colNames = df.columns.tolist()
data = []
for i, colName in enumerate(colNames):
    #print(i)
    print(colName)
    colVals = df[colName].values.flatten()
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

