
##############################################################

#202101-EDA-VDA-Challenge Project-RJC-MSC-DS

# Group-K
# subject-Python & Data Analysis
# Name of Members-
#  Ritvik Kanchan
#  Manoj Yadav
#  Shreya Shinde
#  Apeksha Singh
##############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import networkx as nx
from scipy.stats import ttest_1samp

##############################################################
# Reading file and Storing it as Dataframe
##############################################################
#df = pd.read_csv('C:/Python/challange/diamonds-m.csv')

while(True):
    G=input("\nEnter path of file: ")
    try:
        if((".xls" in G)==True or (".xlsx" in G)==True):
            df = pd.ExcelFile(G)
            l=input("\nSheet name: ")
            df=df.parse(l)
        elif((".fwf" in G)==True):
            df=pd.read_fwf(G)
        elif((".tdll" in G)==True):
            l=input("\nSepration type: ")
            df=pd.read_csv(G, sep=l)
        elif((".csv" in G)==True):
            df=pd.read_csv(G)
        else:
            print("\nFile Did not Found Reason:Invalid path/Filetype")    
            continue

##############################################################
# Structure of the dataset:-
##############################################################\
    
        print("\n*** Structure ***")
        print(df.info())

##############################################################
# Data Type of Each Column:-
##############################################################

        print("\nData Type of Each Column")
        print(df.dtypes)

#############################################################
#Checking if unique identifyer column is there or not if not then creating it
#############################################################

        idc=""
        for g in df.columns:
            K=int(len(df[g]))
            if(str(g).upper()=="ID"):
                idc=g
                flag=True
                break
        if(idc==""):
            df["Id"]=list(range(1,K+1))
        else:
            df=df.rename(columns={g:'Id'})
            
##############################################################
#Length of Alphanumeric Column:-
##############################################################

        max=0
        for g in df.columns:
            print("\n***"+str(g)+"***")
            for r in df[g]:
                r=str(r)
                if(max<len(r)):
                    max=len(r)
            print("\nLength=",max)
            max=0
            
##############################################################
# precision & scale of numeric columns:-
##############################################################
        numericol=[]
        for I in df.columns:
            K=str(df[I].dtype)
            if(("int" in K)==True or ("float" in K)==True):
                if(I!="Id"):
                    numericol.append(str(I))      
        print("\nNumeric columns=",numericol)
        
        if(len(numericol)!=0):
            p=0
            s=0
            for g in numericol:
                print("\n***"+str(g)+"***")
                for r in df[g]: 
                    r=str(r)
                    if("." in r):
                        pe=len(r)-1
                        r=r.split('.')
                        r=r[1]
                        se=len(r)
                    else:
                        pe=len(r)
                        se=0
                    if(pe>p and se>s):
                        p=pe
                        s=se
                print("precision= ",p)
                print("scale= ",s)
                p=0
                s=0
        else:
            print("\nThere are no numeric columns in dataset")

##############################################################
# What are Significant columns ?
#Column which does not represents Id,name and Discription are Significant Columns in Dataframe.
#Identifying significant columns of the data set
##############################################################

        Significantcol=[]
        for I in df.columns:
            K=str(df[I].dtype)
            if(str(I).upper()!="ID" and str(I).upper()!="NAME" and str(I).upper()!="DISCRIPTION"):
                if(("object" in K)==True):
                    if(int(len(df[I].unique()))<int(len(df[I])/9)):
                        Significantcol.append(str(I))
                else:       
                    Significantcol.append(str(I))
        print("\nSignificant columns=",Significantcol)
        
##############################################################
#Number of Nulls and number of Zeros in columns:-
##############################################################

        for g in df.columns:
            colName = g
            print("\n***"+str(g)+"***")
            print("Null Values: ", df[colName].isnull().sum())
            print("Zero Values: ", (df[colName]==0).sum())

##############################################################
#Obvious errors of each Columns:-
# As dataframe can be used in Machine Learning application we have to convert it into numeric categorical value
# for this we will use cat codes
##############################################################

        categorialcol=[]
        for I in df.columns:
            K=str(df[I].dtype)
            if(("object" in K)==True):
                if(int(len(df[I].unique()))>1):
                    categorialcol.append(str(I))
        print("\ncategorial columns=",categorialcol)
        
        for I in categorialcol:    
            print("\n***"+I+"***")
            print(df.groupby([I])['Id'].count())
            print("\nCleaning Data-")
            df[I] = df[I].str.upper()
            df[I]=df[I].str.upper().str.strip()
            print("\nCleaned Data -")
            print(df.groupby([I])['Id'].count())
            print("*Categoric Data -*")
            print(df[I].unique())
            df[I] = pd.Categorical(df[I])
            df[I] = df[I].cat.codes
            print(df[I].unique())
    
##############################################################
#Replace null values with median value of Every numeric column
##############################################################

        for g in numericol:
            print("\n***"+str(g)+"***")
            print("*Original Count*")
            print("Null Values: ",df[g].isnull().sum())
            df[g] = df[g].fillna(df[g].median())
            print("*Cleaned Count*")
            print("Null Values: ",df[g].isnull().sum())    
            
##############################################################
#As there are Extreme values Present in dataset we will not use Mean to replace zero.
#We will Replace Zero values with median value of Every numeric column
##############################################################

        for g in numericol:
            if(int(utils.colOutCount(df[g].values))!=0):
                print("\nReplace Zero values with median value of "+str(g)+" column")
                print("\n***"+str(g)+"***")
                print("*Original Count*")
                print("Zero Values: ", (df[g]==0).sum())
                df[g] = np.where(df[g]==0, df[g].median(), df[g])
                print("*Cleaned Count*")
                print("Zero Values: ", (df[g]==0).sum())
            else:
                print("\nReplace Zero values with mean value of "+str(g)+" column")
                print("\n***"+str(g)+"***")
                print("*Original Count*")
                print("Zero Values: ", (df[g]==0).sum())
                df[g] = np.where(df[g]==0, df[g].mean(), df[g])
                print("*Cleaned Count*")
                print("Zero Values: ", (df[g]==0).sum())
                
##############################################################
#Provide the quartile summary along with the count, mean & sum for each significant column
##############################################################

        for g in numericol:
            print("\n***"+str(g)+"***")
            print("count=",df[g].count())
            print("sum=",df[g].sum())
            print("mean=",df[g].mean())
            print("\nQuertile summary-")
            print("25% =",df[g].quantile(q=0.25))
            print("50% =",df[g].quantile(q=0.5))
            print("75% =",df[g].quantile(q=0.75))

##############################################################
#Provide the range, variance and standard deviation for each significant column
##############################################################

        for g in numericol:
            print("\n***"+str(g)+"***")
            print("range=",df[g].max()-df[g].min())
            print("Variance=",df[g].var())
            print("Standard Deviation=",df[g].std())    

##############################################################
#Provide the count of outliers and their value for each significant column
#Before we Perform code we need to set utils library colOutValues and colOutCount code iqr*3.0 to iqr*1.5
#Set it to show Normal Limit outliers
##############################################################

        for g in numericol:
            print("\n***"+str(g)+"***")
            print('\nOutlier Count-')
            print(utils.colOutCount(df[g].values))    
            print('\nOutlier Values-')
            print(utils.colOutValues(df[g].values))

##############################################################
#class variables
#class variables are categorical variable.
##############################################################

        for I in categorialcol:
            plt.figure()
            sns.countplot(df[I],label="Count")
            plt.title(I)
            plt.show()

##############################################################
#For all numeric columns Provide histogram
##############################################################

        print('\n*** Histograms ***')
        for g in numericol:
            colValues = df[g].values
            plt.figure()
            sns.distplot(colValues, bins=7, kde=False, color='r')
            plt.title(g)
            plt.ylabel(g)
            plt.xlabel('Bins')
            plt.show()

##############################################################
#For all numeric columns Provide box & whisker plot
##############################################################    

        print('\n*** Boxplot ***')
        for g in numericol:
            plt.figure()
            sns.boxplot(y=df[g], color='r')
            plt.title(g)
            plt.ylabel(g)
            plt.xlabel('Bins')
            plt.show()

##############################################################
#For all numeric columns correlation table & graph
##############################################################

        dfo = df[numericol]
        print("\n*** Correlation Table ***")
        pd.options.display.float_format = '{:,.3f}'.format
        df1=dfo.corr()
        print(df1)
        
        plt.figure(figsize=(8,8))
        ax = sns.heatmap(df1, annot=True, cmap="PiYG")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)
        plt.show()

##############################################################
#relationship chart showing relation of each numeric column with all other numeric columns .
##############################################################    
        
        plt.figure(figsize=(20,10))
        nx.draw(g,with_labels=true,node_size=5000,font_size=20)
        ply.show()
##############################################################
#difference between the Actual Depth & Ideal Depth
##############################################################    

        if(("diamonds-m" in G)==True):
            diff=df[['depth']]
            coln=df.loc[:,"x":"y"]
            df['mean']=coln.mean(axis=1)
            depth_P=df['z']/df['mean']
            print(depth_P)
            diff["Dfference"]=df['depth']-depth_P
            print("\n",diff)

##############################################################
#Checking if user want to exit or renter with diffrent dataset
##############################################################

        S=input("\ndo you want to continue (y/n)")    
        if(S!="y"):
            break
    except: 
        print("\nFile Did not Found Reason:Invalid path/Filetype")
   
    

##############################################################

#Summary
# In this project we performed Exploratory Data Analysis (EDA) and Visual Data Analytics (VDA) on Diamonds-m.csv given datasets. 
# A dataset “diamond s-m. csv” containing the prices and other attributes of almost 54,000 diamonds and 10 variables:'carat', 'cut', 'color',
# 'clarity', 'popularity', 'depth', 'table', 'price', 'x', 'y', 'z' 
# We first analysed the structure and datatype of every column. 
# In order to get further better information about column’s we found out length of alphanumeric column’s and precision and scale of numeric columns. 
# After basic analysis we performed EDA by Transformation/Data cleaning of significant column i.e. (handling null, zero values and Obvious errors in significant columns).
# After that we computed basic statistic calculation fields like quartile summary with count, mean, sum, range, variance and standard deviation then we provided count of outliers and their value in each significant column.
# After EDA we performed VDA by providing frequency distribution table & chart for class variables, histogram for all numeric columns, box & whisker plot for numeric variables, correlation table & graph(heatmap) . 

##############################################################

