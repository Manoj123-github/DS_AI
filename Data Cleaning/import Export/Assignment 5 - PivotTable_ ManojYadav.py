# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:43:15 2020

@author: Manoj Yadav
"""
import pandas as pd
import numpy as np

df = pd.read_csv("C:/Python/assignment/A04B-Pandas-Sales.csv")
print(df.columns)
print(df.head())


pd.pivot_table(df, index=["Customer Name"], values=["Sales","Costs","Profit"], aggfunc=[np.sum])



# 01 Year wise Sales, Costs, Profits
dfp=df.copy()
dfp['Order Year']= dfp["Order Date"].str[:4]
pd.pivot_table(dfp, index=["Order Year"], values=["Sales","Costs","Profit"])


#02 Customer Segment wise Sales, Costs, Profits

pd.pivot_table(df, index=['Customer Name','Segment'], values=["Sales","Costs","Profit"])


#03 Product Category wise Sales, Costs, Profits

pd.pivot_table(df, index=['Category'], values=["Sales","Costs","Profit"])


#04 Product Sub-Category wise Sales, Costs, Profits

 pd.pivot_table(df, index=['Sub-Category'], values=["Sales","Costs","Profit"])
 
 
 #05 Customer Segment & Product Category wise Sales, Costs, Profits
 
 pd.pivot_table(df, index=['Segment','Category'], values=["Sales","Costs","Profit"])
 
 
 #06 Product Category & Product Sub-Category wise Sales, Costs, Profits
 
 pd.pivot_table(df, index=['Category','Sub-Category'], values=["Sales","Costs","Profit"])
 
 
 #07 Region, State, City  wise Sales, Costs, Profits
 
 pd.pivot_table(df, index=['Region','State','City'], values=["Sales","Costs","Profit"])
 
 
 # Q8. Show Top 10 Customer Name by Sales
 pd.pivot_table(df, index=["Customer Name",], values=["Sales"]).head(10)

 
 
 # Q9. Show Top 10 Product Name by Profit
 pd.pivot_table(df, index=["Product Name",], values=["Profit"]).head(10)
 
 
 
 # Q10.Count of orders based on ShipMode
 print(df.groupby(['Ship Mode']).count())