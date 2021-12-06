

# hides all warnings
import warnings
warnings.filterwarnings('ignore')

# imports
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 5)
# sns
import seaborn as sns
# util
import utils


##############################################################
# Read Data 
##############################################################

# read dataset
df = pd.read_csv('./data/naive-bayes.csv')


##############################################################
# Exploratory Data Analytics
##############################################################

# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())


##############################################################
# Class Variable & Counts
##############################################################

# store class variable  
# change as required
clsVars = "Play"
print("\n*** Class Vars ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# get unique Species names
print("\n*** Unique Species - Categoric Alpha***")
lnLabels = df[clsVars].unique()
print(lnLabels)


##############################################################
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
print("None ...")

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers if required

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# check mean
print('\n*** Mean In Columns ***')
print(df.mean())

# handle normalization if required

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required

##############################################################
# Frequency Tables
##############################################################

# print freq tables
print('\n*** Frequency Table ***')
# outlook
print('\n*Outlook*')
ftOutlook = pd.crosstab(df['Outlook'], df['Play'], margins = True)
print(ftOutlook)
# temp
print('\n*Temp*')
ftTemp    = pd.crosstab(df['Temp'], df['Play'], margins = True)
print(ftTemp)
# humidity
print('\n*Humidity*')
ftHumid   = pd.crosstab(df['Humid'], df['Play'], margins = True)
print(ftHumid)
# wind
print('\n*Wind*')
ftWind   = pd.crosstab(df['Wind'], df['Play'], margins = True)
print(ftWind)
# play
print('\n*Play*')
ftPlay    = pd.crosstab(df['Play'],"count", margins=True)
print(ftPlay)

##############################################################
# access individual cell in FreqTable
##############################################################
# outlook "Sunny" & Play "Yes"
print(ftOutlook)
print("")
print("Sunny & Yes: ",ftOutlook.loc['Sunny']['Yes'])
print("Play = Yes: ",ftPlay.loc['Yes'][0])
# probability "Sunny" & Play "Yes"
print("P(Outlook=Sunny|Play=yes):",ftOutlook.loc['Sunny']['Yes']/ftPlay.loc['Yes'][0])

##############################################################
# Predict
# Outlook	Temp	Humid	Wind
##############################################################
# create dataframe
data = {'Outlook': ['Cloudy','Sunny'],
        'Temp'   : ['Cool', 'Hot'], 
        'Humid'  : ['High','High'],
        'Wind'   : ['Strong','Strong']}
dfp = pd.DataFrame(data)
print(dfp)

# getPredict
def getPredict(pIndex):
    #pIndex = 0
    print(pIndex)
    # process based on TrnxNumb
    vOutlook = dfp.Outlook[pIndex]
    vTemp    = dfp.Temp[pIndex]
    vHumid   = dfp.Humid[pIndex]
    vWind    = dfp.Wind[pIndex]
    # prob of Outlook in Yes & No
    print('\n*Outlook*')
    print(vOutlook)
    print(ftOutlook)
    print(ftPlay)
    vPofOlookInYes = ftOutlook.loc[vOutlook]['Yes']/ftPlay.loc['Yes'][0]
    vPofOlookInNo  = ftOutlook.loc[vOutlook]['No']/ftPlay.loc['No'][0]
    print("P(Outlook="+vOutlook+"|Play=Yes):",vPofOlookInYes)
    print("P(Outlook="+vOutlook+"|Play=No) :",vPofOlookInNo)
    # prob of Temp in Yes & No
    print('\n*Temp*')
    print(vTemp)
    print(ftTemp)
    print(ftPlay)
    vPofTempInYes = ftTemp.loc[vTemp]['Yes']/ftPlay.loc['Yes'][0]
    vPofTempInNo  = ftTemp.loc[vTemp]['No']/ftPlay.loc['No'][0]
    print("P(Temp="+vTemp+"|Play=Yes):",vPofTempInYes)
    print("P(Temp="+vTemp+"|Play=No) :",vPofTempInNo)
    # prob of Humid in Yes & No
    print('\n*Humidity*')
    print(vHumid)
    print(ftHumid)
    print(ftPlay)
    vPofHumidInYes = ftHumid.loc[vHumid]['Yes']/ftPlay.loc['Yes'][0]
    vPofHumidInNo  = ftHumid.loc[vHumid]['No']/ftPlay.loc['No'][0]
    print("P(Humid="+vHumid+"|Play=Yes):",vPofHumidInYes)
    print("P(Humid="+vHumid+"|Play=No) :",vPofHumidInNo)
    # prob of Wind in Yes & No
    print('\n*Wind*')
    print(vWind)
    print(ftWind)
    print(ftPlay)
    vPofWindInYes = ftWind.loc[vWind]['Yes']/ftPlay.loc['Yes'][0]
    vPofWindInNo  = ftWind.loc[vWind]['No']/ftPlay.loc['No'][0]
    print("P(Wind="+vWind+"|Play=Yes):",vPofWindInYes)
    print("P(Wind="+vWind+"|Play=No) :",vPofWindInNo)
    # probability of Play Yes & No
    vPofPlayYes = vPofOlookInYes * vPofTempInYes * vPofHumidInYes * vPofWindInYes
    vPofPlayNo  = vPofOlookInNo  * vPofTempInNo  * vPofHumidInNo  * vPofWindInNo
    print("P(Play=Yes):",vPofPlayYes)
    print("P(Play=No) :",vPofPlayNo)
    # 
    if vPofPlayYes >= vPofPlayNo:
        vRetVals = "Yes"
    else:
        vRetVals = "No"
    print(vRetVals)
    
    return(vRetVals)


# iterate
dfp['Predict'] = ''
dfp['Predict'] = [getPredict(i) for i in dfp.index]
print(dfp)
