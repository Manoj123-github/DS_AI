# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 00:56:20 2020
@filename: utils.py
@describe: utility functions
@dataset: Nil
@author: cyruslentin
"""
import numpy as np
import pandas as pd

# returns: number of rows which contain <blank>
# usage: colSpaceCount(colName)
def colSpaceCount(colName):
	return (colName.str.strip().values == '').sum()

# returns: number of rows which contain <blank> iterating through each col of df
# usage: SpaceCount(df)
def SpaceCount(df): 
    colNames = df.columns
    strRetValue = ""
    for colName in colNames:
        #if type(colName) == "object":
        if df[colName].dtype == "object": 
            #print(colName)
            spcCount = colSpaceCount(df[colName])
            #print(spcCount)
            strRetValue = strRetValue + colName.ljust(15, ' ') + "   " + str(spcCount) + "\n"
    return(strRetValue)

# returns: count of outliers in the colName
# usage: colOutCount(colValues)
def colOutCount(colValues):
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 3.0)
    upper_bound = quartile_3 + (iqr * 3.0)
    ndOutData = np.where((colValues > upper_bound) | (colValues < lower_bound))
    ndOutData = np.array(ndOutData)
    return ndOutData.size

# returns: count of outliers in each column of dataframe
# usage: OutlierCount(df): 
def OutlierCount(df): 
    colNames = df.columns
    strRetValue = ""
    for colName in colNames:
        #print(colName)
        colValues = df[colName].values
        #print(colValues)
        outCount = colOutCount(colValues)
        #print(outCount)
        strRetValue = strRetValue + colName.ljust(15, ' ') + "   " + str(outCount) + "\n"
    return(strRetValue)

# returns: row index in the colName
# usage: colOutIndex(colValues)
def colOutIndex(colValues):
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 3.0)
    upper_bound = quartile_3 + (iqr * 3.0)
    ndOutData = np.where((colValues > upper_bound) | (colValues < lower_bound))
    ndOutData = np.array(ndOutData)
    return ndOutData

# returns: row index of outliers in each column of dataframe
# usage: OutlierIndexs(df): 
def OutlierIndex(df): 
    colNames = df.columns
    strRetValue = ""
    for colName in colNames:
        if df[colName].dtype == "object": 
            continue
        colValues = df[colName].values
        #print('Column: ', colName)
        strRetValue = strRetValue + colName + " " + "\n"
        strRetValue = strRetValue + str(colOutIndex(colValues)) + " \n"
        #print(outValues)
        #print(" ")
    return(strRetValue)

# returns: actual outliers values in the colName
# usage: colOutValues(colValues)
def colOutValues(colValues):
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 3.0)
    upper_bound = quartile_3 + (iqr * 3.0)
    ndOutData = np.where((colValues > upper_bound) | (colValues < lower_bound))
    ndOutData = np.array(colValues[ndOutData])
    return ndOutData

# returns: actual of outliers in each column of dataframe
# usage: OutlierValues(df): 
def OutlierValues(df): 
    colNames = df.columns
    strRetValue = ""
    for colName in colNames:
        if df[colName].dtype == "object": 
            continue
        colValues = df[colName].values
        print('Column: ', colName)
        strRetValue = strRetValue + colName + " " + "\n"
        strRetValue = strRetValue + str(colOutValues(colValues)) + " \n"
        print(colOutValues(colValues))
        print(" ")
    return(strRetValue)


# returns: upper boud & lower bound for array values or df[col].values 
# usage: OutlierLimits(df[col].values): 
def OutlierLimits(colValues): 
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 3.0)
    upper_bound = quartile_3 + (iqr * 3.0)
    return lower_bound, upper_bound

# standardize data - all cols of df will be Standardized except colClass 
# x_scaled = (x — mean(x)) / stddev(x)
# all values will be between 1 & -1
# Usage: StandardizeData(df, colClass) 
# df datarame, colClass - col to ignore while transformation  
def StandardizeData(df, colClass):
        # preparing for standadrising
        colNames = df.columns.tolist()
        lstClass = df[colClass]
        # standardizaion : 
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # fit
        ar = scaler.fit_transform(df)
        # transform
        df = pd.DataFrame(data=ar)
        # # change as required
        df.columns = colNames
        df[colClass] = lstClass
        return(df)

# normalize data - all cols of df will be Normalized except colClass 
# x_scaled = (x-min(x)) / (max(x)–min(x))
# all values will be between 0 & 1
# Usage: NormalizeeData(df, colClass) 
# df datarame, colClass - col to ignore while transformation  
def NormalizeData(df, colClass):
        # preparing for standadrising
        colNames = df.columns.tolist()
        lstClass = df[colClass]
        from sklearn.preprocessing import MinMaxScaler
        # normalizing the data
        scaler = MinMaxScaler()
        # fit
        ar = scaler.fit_transform(df)
        # transform
        df = pd.DataFrame(data=ar)
        # # change as required
        df.columns = colNames
        df[colClass] = lstClass
        return(df)


# MaxAbsScaled data - all cols of df will be MaxAbsScaled except colClass 
# x_scaled = x / max(abs(x))
# Usage: MaxAbsScaledData(df, colClass) 
# df datarame, colClass - col to ignore while transformation  
def MaxAbsScaledData(df, colClass):
        # preparing for standadrising
        colNames = df.columns.tolist()
        lstClass = df[colClass]
        # normalizing the data 
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        # fit
        ar = scaler.fit_transform(df)
        # transform
        df = pd.DataFrame(data=ar)
        # # change as required
        df.columns = colNames
        df[colClass] = lstClass
        return(df)


# getFeatureScoresXTC - Extra Tree Classifier
# prints feature scores of all cols except colClass 
# Usage: getFeatureScoresXTC(df, colClass) 
# df datarame, colClass - col to ignore while transformation  
def getFeatureScoresXTC(df, colClass):
    # make into array
    #print("\n*** Prepare Data ***")
    # store class variable  ... change as required
    clsVars = colClass
    allCols = df.columns.tolist()
    #print(allCols)
    allCols.remove(clsVars)
    #print(allCols)
    # split into X & y        
    X = df[allCols].values
    y = df[clsVars].values

    # feature extraction with ExtraTreesClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    # extraction
    model = ExtraTreesClassifier(n_estimators=10, random_state=707)
    model.fit(X, y)
    #print("\n*** Column Scores ***")
    # summarize scores
    np.set_printoptions(precision=3)
    #print(model.feature_importances_)
    # data frame
    dfm =  pd.DataFrame({'Cols':allCols, 'Imp':model.feature_importances_})  
    dfm.sort_values(by='Imp', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
    print(dfm)
    # remove dfm from mem
    del dfm
    return

# getFeatureScoresSKB - Select K Best
# prints feature scores of all cols except colClass 
# Usage: getFeatureScoresXTC(df, colClass) 
# df datarame, colClass - col to ignore while transformation  
def getFeatureScoresSKB(df, colClass):
    # make into array
    #print("\n*** Prepare Data ***")
    # store class variable  ... change as required
    clsVars = colClass
    allCols = df.columns.tolist()
    #print(allCols)
    allCols.remove(clsVars)
    #print(allCols)
    # split into X & y        
    X = df[allCols].values
    y = df[clsVars].values
    
    # Feature extraction with selectBest
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    # feature extraction
    model = SelectKBest(score_func=f_classif, k=4)
    fit = model.fit(X, y)
    # summarize scores
    np.set_printoptions(precision=3)
    #print(fit.scores_)
    # data frame
    dfm =  pd.DataFrame({'Cols':allCols, 'Imp':fit.scores_})  
    dfm.sort_values(by='Imp', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
    print(dfm)




#!pip install -U imbalanced-learn
# https://pypi.org/project/imbalanced-learn/
# Random Over Sampler ... creates duplicate records of the lower sample
# Usage: getOverSamplerData(X, y) ... requires standard X, y 
def getOverSamplerData(X,y): 
    # import
    from imblearn.over_sampling import RandomOverSampler
    # create os object
    os =  RandomOverSampler(random_state = 707)
    # generate over sampled X, y
    return (os.fit_sample(X, y))

#!pip install -U imbalanced-learn
# https://pypi.org/project/imbalanced-learn/
# SMOTE - Synthetic Minority Oversampling Technique - creates random new synthetic records
# Usage: getSmoteSamplerData(X, y) ... requires standard X, y 
def getSmoteSamplerData(X,y): 
    # import
    from imblearn.over_sampling import SMOTE
    # create smote object
    sm = SMOTE(random_state = 707)
    # generate over sampled X, y
    return (sm.fit_resample(X, y))

#!pip install -U imbalanced-learn
# https://pypi.org/project/imbalanced-learn/
# Random Under Sampler ... deleted records of the higher sample
# Usage: getUnderSamplerData(X, y) ... requires standard X, y 
def getUnderSamplerData(X,y): 
    # import
    from imblearn.under_sampling import RandomUnderSampler
    # create os object
    us =  RandomUnderSampler(random_state = 707, replacement=True)
    # generate over sampled X, y
    return (us.fit_resample(X, y))


