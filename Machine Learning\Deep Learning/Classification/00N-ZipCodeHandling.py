
# pandas
import pandas as pd
# numpy
import numpy as np
# real exp
import re

# reading df file
df = pd.read_csv("./data/us-officials.csv")

# view Col Info
print("\n*** Info ***")
print(df.info())

# keep only ID & OrderDate Cols
df = df[['Candidate Name','City','State','Zip Code','Phone']]

# view Col Info
print("\n*** Info ***")
print(df.info())

# validate zipCode function
def validateZipcode(pZipCode, pFormat="USS"):
    #print(pZipCode)
    # US short zip (5 digit)
    if pFormat == "USS":
        pFormat = "[1-9]{1}\d{4}"
    # US long zip (5-4 digit)
    elif pFormat == "USL":
        pFormat = "[1-9]{1}\d{4}-\d{4}"
    # US short & long zip (5 digit or 5-4 digit)
    elif pFormat == "USSL":
        pFormat = "US Short & Long"
        pFormat1 = "[1-9]{1}\d{4}"
        pFormat2 = "[1-9]{1}\d{4}-\d{4}"
    # India zip (6 digit)
    elif pFormat == "IN":
        pFormat = "[1-9]{1}\d{5}"
    # UK Zip - as supplied by UK Govt
    # https://stackoverflow.com/questions/164979/regex-for-matching-uk-postcodes
    elif pFormat == "UK":
        pFormat = "([Gg][Ii][Rr] 0[Aa]{2})|((([A-Za-z][0-9]{1,2})|(([A-Za-z][A-Ha-hJ-Yj-y][0-9]{1,2})|(([A-Za-z][0-9][A-Za-z])|([A-Za-z][A-Ha-hJ-Yj-y][0-9][A-Za-z]?))))\s?[0-9][A-Za-z]{2})"
    #print(pFormat)
    try:
        if pFormat=="USSL":
            if re.search(pFormat1, pZipCode) or re.search(pFormat2, pZipCode):
                vRets = "Valid"
            else:
                vRets = "Invalid"
        else:
            if re.search(pFormat, pZipCode):
                vRets = "Valid"
            else:
                vRets = "Invalid"
    except:
        vRets = "Invalid"
    #print(vRets)    
    return vRets

# validate zipcode
df["ValidZipcodes"] = [validateZipcode(z,'USSL') for z in df['Zip Code']]
if ("Invalid" in df['ValidZipcodes']):
    print("Column contains Invalid Zips")
else:
    print("Column contains all Valid Zips")

# convert US-LongZip to number
df["Zip Code"] = df["Zip Code"].str.replace("-","")
if df["Zip Code"].isnull().sum() > 0:
    df["Zip Code"] = df["Zip Code"].astype(np.float64)
else:
    df["Zip Code"] = df["Zip Code"].astype(np.int64)
# view Col Info
print("\n*** Info ***")
print(df.info())

# validate phone function
def validatePhone(pPhonNumb, pFormat="US"):
    #print(pZipCode)
    # US phone number
    if pFormat == "US":
        pFormat = "[1-9]{1}\d{2}-\d{3}-\d{4}"
    # India mobile    
    if pFormat == "IN-M":
        pFormat = "[7-9]{1}[0-9]{9}"
    # India landline with std code
    if pFormat == "IN-L":            
        pFormat = "[2-6]{1}[0-9]{9}"
    try:
        # check
        if re.search(pFormat, pPhonNumb):
            vRets = "Valid"
        else:
            vRets = "Invalid"
    except:
        vRets = "Invalid"
    #print(vRets)    
    return vRets

# validate phone 
df["ValidPhones"] = [validatePhone(p) for p in df['Phone']]
if ("Invalid" in df['ValidPhones']):
    print("Column contains Invalid Phones")
else:
    print("Column contains all Valid Phones")

# validate email function
def validateEmail(pEmail):
    #print(pEmail)
    # regex for validating an Email
    pFormat = '^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'    
    try:
        # check
        if re.search(pFormat, pEmail):
            vRets = "Valid"
        else:
            vRets = "Invalid"
    except:
        vRets = "Invalid"
    #print(vRets)    
    return vRets

lEmails = ['clentin@gmail.com','invalid@com','cyrus.lentin@gmail.com','cyrus@lentins.co.in', 'cyrus.lentin@lentins.co.in']
lValidEmails = [validateEmail(e) for e in lEmails]
print(lEmails)
print(lValidEmails)
if ("Invalid" in lValidEmails):
    print("Column contains Invalid Emails")
else:
    print("Column contains all Valid Emails")

# structure of address in US
# AddrLine-1, AddrLine-2, City, State, Zip    
# 5d zips are city wise 5d-4d zips are city+area zips
# so if city is same at least 5d zips will match

# structure of address in India
# AddrLine-1(Flat/Bldg), AddrLine-2(Road Name), AddrLine-3(Area Name), City, State, Zip    
# 6d zips are area-city wise
# so if area&city is same at least 6d zips will match

#dfx = df[ df['Zip Code'].isnull() ]
#dfy = df[ df['City']=='Natalbany' ]

# update missing zip function
def updateMissingZips(pIndex):
    #pIndex=5131
    #print(pIndex)
    vCity = df['City'][pIndex]
    #print(vCity)
    dft = df[ df['City'] == vCity ]
    dft = dft[ dft['Zip Code'].notnull() ]
    #print(dft)
    if len(dft.index) >= 1:
        vZipCode = dft.iloc[0]['Zip Code']
        #print(type(vZipCode))
        if type(vZipCode)=="str":
            vZipCode = vZipCode[0:5]
        df['Zip Code'][pIndex] = vZipCode
        #print(vZipCode)
    else:
        vZipCode = None
    return vZipCode

# find missing zip
dfTmp = df[df['Zip Code'].isnull()]
lZipCode = [updateMissingZips(i) for i in dfTmp.index]
print(lZipCode)

# find missing city

