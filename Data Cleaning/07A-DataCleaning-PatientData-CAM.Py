
# import
import pandas as pd
import numpy as np
import string as st
#import utils

#reading data file
df = pd.read_csv("./patient-data.csv")
# view Col Info
print(df.info())
# view Col Info
print(df.describe())
# view Data
print(df.head())

#reading data file
uss = pd.read_csv("./us-states.csv")
# view Col Info
print(uss.info())
# view Col Info
print(uss.describe())
# view Data
print(uss.head())

# is valid state functions
def isValidState(pState):
    global uss

    # filter state file bease sName param
    dft = uss[ uss['State']==pState ]
    #dft
    vRets=""
    vState=""
    try:
        #print("In Try")
        vState = dft.iloc[0,0]
        vRets = "Valid"
    except:    
        #print("In Exc")
        vState = ""
        vRets = "Invalid"
    return (vRets)

# get isValidState for all rows
print('Checking State Validity ...')
df['ValidState'] = [isValidState(s) for s in df.State]
print('Checking State Validity ... Completed OK')










