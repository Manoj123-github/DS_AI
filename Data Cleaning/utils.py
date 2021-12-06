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

# space count per coulmn
def colSpaceCount(colName):
    """
    returns: 
        number of rows which contain <blank>
    usage: 
        colSpaceCount(colName)
    """ 
    return (colName.str.strip().values == '').sum()


# space count for data frame
def SpaceCount(df): 
    """
    returns:  
        number of rows which contain <blank> iterating through each col of df
    usage: 
        SpaceCount(df)
    """
    colNames = df.columns
    dsRetValue = pd.Series() 
    for colName in colNames:
        if df[colName].dtype == "object": 
            dsRetValue[colName] = colSpaceCount(df[colName])
    return(dsRetValue)


# outlier count for column
def colOutCount(colValues):
    """
    returns: 
        count of outliers in the colName
    usage: 
        colOutCount(colValues)
    """
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 3.0)
    upper_bound = quartile_3 + (iqr * 3.0)
    ndOutData = np.where((colValues > upper_bound) | (colValues < lower_bound))
    ndOutData = np.array(ndOutData)
    return ndOutData.size


# outlier count for dataframe
def OutlierCount(df): 
    """
    returns: 
        count of outliers in each column of dataframe
    usage: 
        OutlierCount(df): 
    """
    colNames = df.columns
    dsRetValue = pd.Series() 
    for colName in colNames:
        if (df[colName].dtypes == 'object'):
            continue
        #print(colName)
        colValues = df[colName].values
        #print(colValues)
        #outCount = colOutCount(colValues)
        #print(outCount)
        dsRetValue[colName] = colOutCount(colValues)
    return(dsRetValue)


# oulier index for column
def colOutIndex(colValues):
    """
    returns: 
        row index in the colName
    usage: 
        colOutIndex(colValues)
    """
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 3.0)
    upper_bound = quartile_3 + (iqr * 3.0)
    ndOutData = np.where((colValues > upper_bound) | (colValues < lower_bound))
    ndOutData = np.array(ndOutData)
    return ndOutData


# oulier index for data frame
def OutlierIndex(df): 
    """
    returns: 
        row index of outliers in each column of dataframe
    usage: 
        OutlierIndex(df): 
    """
    colNames = df.columns
    dsRetValue = pd.Series() 
    for colName in colNames:
        if (df[colName].dtypes == 'object'):
            continue
        colValues = df[colName].values
        dsRetValue[colName] = str(colOutIndex(colValues))
    return(dsRetValue)


# outlier values for column 
def colOutValues(colValues):
    """
    returns: 
        actual outliers values in the colName
    usage: 
        colOutValues(colValues)
    """
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 3.0)
    upper_bound = quartile_3 + (iqr * 3.0)
    ndOutData = np.where((colValues > upper_bound) | (colValues < lower_bound))
    ndOutData = np.array(ndOutData)
    return ndOutData


# outlier values for dataframe 
def OutlierValues(df): 
    """
    returns: 
        actual of outliers in each column of dataframe
    usage: 
        OutlierValues(df): 
    """
    colNames = df.columns
    dsRetValue = pd.Series() 
    for colName in colNames:
        if (df[colName].dtypes == 'object'):
            continue
        colValues = df[colName].values
        #print('Column: ', colName)
        #strRetValue = strRetValue + colName + " " + "\n"
        #strRetValue = strRetValue + str(colOutValues(colValues)) + " \n"
        #print(colOutValues(colValues))
        #print(" ")
        dsRetValue[colName] = str(colOutValues(colValues))
    return(dsRetValue)


# outlier limits
def OutlierLimits(colValues): 
    """
    returns: 
        upper boud & lower bound for array values or df[col].values 
    usage: 
        OutlierLimits(df[col].values): 
    """
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 3.0)
    upper_bound = quartile_3 + (iqr * 3.0)
    return lower_bound, upper_bound


# standardize data
def StandardizeData(df, colClass):
    """
    desc:
        standardize data - all cols of df will be Standardized except colClass 
        x_scaled = (x — mean(x)) / stddev(x)
        all values will be between 1 & -1
    usage: 
        StandardizeData(df, colClass) 
    params:
        df datarame, colClass - col to ignore while transformation  
    """
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


# normalize data
def NormalizeData(df, colClass):
    """
    desc:
        normalize data - all cols of df will be Normalized except colClass 
        x_scaled = (x-min(x)) / (max(x)–min(x))
        all values will be between 0 & 1
    usage: 
        NormalizeeData(df, colClass) 
    params:
        df datarame, colClass - col to ignore while transformation  
    """
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


# Max Abs Scalaed Data
def MaxAbsScaledData(df, colClass):
    """
    desc:
        MaxAbsScaled data - all cols of df will be MaxAbsScaled except colClass 
        x_scaled = x / max(abs(x))
    Usage: 
        MaxAbsScaledData(df, colClass) 
    Params:
        df datarame, colClass - col to ignore while transformation  
    """
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
def getFeatureScoresXTC(df, colClass):
    """
    desc:
        prints feature scores of all cols except colClass 
    usage: 
        getFeatureScoresXTC(df, colClass) 
    params:
        df datarame, colClass - col to ignore while transformation  
   """
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
    return (dfm)


# getFeatureScoresSKB - Select K Best
def getFeatureScoresSKB(df, colClass):
    """
    desc:
        prints feature scores of all cols except colClass 
    usage: 
        getFeatureScoresXTC(df, colClass) 
    params:
        df datarame, colClass - col to ignore while transformation  
    """
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
    return (dfm)


# get OverSampleData
def getOverSamplerData(X,y): 
    """
    install:
        !pip install -U imbalanced-learn
    url:
        https://pypi.org/project/imbalanced-learn/
    desc:
        Random Over Sampler ... 
        creates duplicate records of the lower sample
        to match the sample size of highest size class
    usage: 
        getOverSamplerData(X, y) ... requires standard X, y 
    """
    # import
    from imblearn.over_sampling import RandomOverSampler
    # create os object
    os =  RandomOverSampler(random_state = 707)
    # generate over sampled X, y
    return (os.fit_resample(X, y))


# get SMOTE Sampler Data
def getSmoteSamplerData(X,y): 
    """
    install:
        !pip install -U imbalanced-learn
    url:
        https://pypi.org/project/imbalanced-learn/
    desc:
        SMOTE - Synthetic Minority Oversampling Technique 
        creates random new synthetic records
        to match the sample size of highest size class
    usage: 
        getSmoteSamplerData(X, y) ... requires standard X, y 
    """
    # import
    from imblearn.over_sampling import SMOTE
    # create smote object
    sm = SMOTE(random_state = 707)
    # generate over sampled X, y
    return (sm.fit_resample(X, y))


# get UnderSamplerData
def getUnderSamplerData(X,y): 
    """
    install:
        !pip install -U imbalanced-learn
    url:
        https://pypi.org/project/imbalanced-learn/
    desc:
        Random Under Sampler ... 
        deletes records of the higher sample
        to match the sample size of lowest size class
    usage:  
        getUnderSamplerData(X, y)
    params:
        requires standard X, y 
    """
    # import
    from imblearn.under_sampling import RandomUnderSampler
    # create os object
    us =  RandomUnderSampler(random_state = 707, replacement=True)
    # generate over sampled X, y
    return (us.fit_resample(X, y))


# one hot encoding
def oheBind(pdf, encCol):
    """
    desc:
        One Hot Encoding 
        Col With Categoric Values A & B is converted to ColA & ColB with 0s & 1s
    usage: 
        oheBind(pdf, encCol)
    params:
        pdf - data frame, encCol - column to be encoded
    returns:
        df with oheCols & encCol deleted
    """
    ohe = pd.get_dummies(pdf[[encCol]])
    #ohe.columns = pdf[encCol].unique()
    rdf = pd.concat([pdf, ohe], axis=1)
    rdf = rdf.drop(encCol, axis=1)
    return(rdf)