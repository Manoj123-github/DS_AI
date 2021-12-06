
# import pandas
import pandas as pd
import xml.etree.cElementTree as et

# function 
# return node text or None 
def getvalueofnode(node):
    return node.text if node is not None else None

# assign spreadsheet filename: file
oXML = et.parse('C:\Python\data\import export data\Catalog.xml')
colNames = ['TITLE', 'ARTIST', 'COUNTRY', 'COMPANY','PRICE','YEAR']
df = pd.DataFrame(columns=colNames)
for node in oXML.getroot():
    TITLE = node.find('TITLE')
    ARTIST = node.find('ARTIST')
    COUNTRY = node.find('COUNTRY')
    COMPANY = node.find('COMPANY')
    PRICE = node.find('PRICE')
    YEAR = node.find('YEAR')
    df = df.append(pd.Series([getvalueofnode(TITLE), getvalueofnode(ARTIST), 
                              getvalueofnode(COUNTRY), getvalueofnode(COMPANY), 
                              getvalueofnode(PRICE), getvalueofnode(YEAR)], 
                            index=colNames),
                            ignore_index=True)
 # columns
print(df.columns) 

 # info 
print(df.info()) 

# summary
print(df.describe()) 

# head
print(df.head())

 