

# import pandas
import pandas as pd
import json

with open('C:\Python\data\import export data\Catalog.json') as jsonFile:  
    data = json.load(jsonFile)
    for p in data['Catalog']:
        print('TITLE: ' + p['TITLE'])
        print('ARTIST: ' + p['ARTIST'])
        print('COUNTRY: ' + p['COUNTRY'])
        print('COMPANY: ' + p['COMPANY'])
        print('PRICE: ' + p['PRICE'])
        print('YEAR: ' + p['YEAR'])
        print('')

# create empty dataframe
df = pd.DataFrame(columns=['TITLE','ARTIST','COUNTRY','COMPANY','PRICE','YEAR'])

# read jason file
with open('C:\Python\data\import export data\Catalog.json') as jsonFile:  
    data = json.load(jsonFile)
    for p in data['Catalog']:
        dft= pd.DataFrame({'TITLE':  p['TITLE'],  'ARTIST': p['ARTIST'], 
                           'COUNTRY':  p['COUNTRY'], 'COMPANY': p['COMPANY'],  
                           'PRICE':  p['PRICE'], 'YEAR': p['YEAR']}, index=[0])
        df = df.append(dft, ignore_index=True)
df = df[['TITLE','ARTIST','COUNTRY','COMPANY','PRICE','YEAR']]
    
# columns
print(df.columns) 

 # info 
print(df.info()) 

# summary
print(df.describe()) 

# head
print(df.head())

 