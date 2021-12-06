

# import
import pandas as pd
import numpy as np
import string as st
#import utils

#reading data file
df = pd.read_csv("./patient-data.csv")

# view Col Info
print("\n*** Info ***")
print(df.info())

# rename Class to InsStatus
df=df.rename(columns = {'Class':'InsStatus'})
print(df.info())

# view Col Info
print("\n*** Summary ***")
print(df.describe())

# view Data
print("\n*** Head ***")
print(df.head())


##################
# checking
##################

# check nulls
print('\n*** Columns With Nulls ***')
df.isnull().sum() 

# check zeros
print('\n*** Columns With Zero ***')
print((df==0).sum())

print('\n*** Columns With Spaces ***')
print(utils.SpaceCount(df))

print(df['ID'].dtype)

# check ID
print("*** ID ***")
print("Null Values")
print(df['ID'].isnull().sum())
print("")

# check Name
print("*** Name ***")
print("Only Alpha Values", df.Name.map(lambda v: v.isalpha()).sum())
print("")

# check Race
print("*** Race ***")
print(df.groupby(['Race'])['Name'].count())
print("")

# check Gender
print("*** Gender ***")
print(df.groupby(['Gender'])['Name'].count())
print("")

# check Smokes
print("*** Smokes ***")
print(df.groupby(['Smokes'])['Name'].count())
print("Data Type")
print(df.Smokes.dtype.name)
print("")

# check HeightInCms
print("*** HeightInCms ***")
print("Outlier Count")
print(utils.colOutCount(df.HeightInCms))
print("Outlier Value")
print(utils.colOutValues(df.HeightInCms))
print("HeightInCms Is < 160")
print(df.HeightInCms[df.HeightInCms<160].count())
print("HeightInCms Is > 180")
print(df.HeightInCms[df.HeightInCms>=180].count())
print("")

# check Height
print("*** WeightInKgs ***")
print("Null Values ")
print(df.WeightInKgs.isnull().sum())
print("Outlier Count")
print(utils.colOutCount(df.WeightInKgs))
print("Outlier Value")
print(utils.colOutValues(df.WeightInKgs))
print("WeightInKgs Is < 65")
print(df.WeightInKgs[df.WeightInKgs<65].count())
print("WeightInKgs Is > 85")
print(df.WeightInKgs[df.WeightInKgs>=85].count())
print("")

# check BirthDate
print("*** BirthDate ***")
print(type(df.BirthDate[0]))
print("")

# check State
print("*** State ***")
print(df.groupby(['State'])['Name'].count())
print("")

# check Pet
print("*** Pet ***")
print(df.groupby(['Pet'])['Name'].count())
print("")

# check health
print("*** Health ***")
print(df.groupby(['Health'])['Name'].count())
print("")

# check InsStatus
print("*** InsStatus ***")
print(df.groupby(['InsStatus'])['Name'].count())
print("")

# check Died
print("*** Died ***")
print(df.groupby(['Died'])['Name'].count())
print("")

# check dates
print("*** RecDate ***")
print(type(df.RecDate[0]))
print("")

##################
# cleaning
##################

#df = pd.read_csv("./data/patient-data.csv")

# get cleanName - Function
def getCleanName(pName):
    # print arg
    #print(pName)
    # clean
    vRetVals = pName.translate(str.maketrans(" ", " ", st.punctuation))
    vRetVals = vRetVals.strip()
    return(vRetVals)  

# get cleanName - Unit Test
def getCleanNameTest():
    x = getCleanName("This, is. a; test: string? check!")
    print(x)


# clean ID
print("*** ID ***")
print("Null Values")
print(df['ID'].isnull().sum())
print("Spaces")
print(utils.colSpaceCount(df['ID']))
print("")

# clean Name
print("*** Name ***")
print("*Original Count*")
print("Not Null Values")
print(df.Name.notnull().sum())
print("Only Alpha Values")
print(df.Name.map(lambda v: v.isalpha()).sum())
df['Name'] = [getCleanName(n) for n in df.Name]
print("")
print("*Cleaned Count*")
print("Not Null Values")
print(df.Name.notnull().sum())
print("Only Alpha Values")
print(df.Name.map(lambda v: v.isalpha()).sum())

# clean Race
print("*** Race ***")
print("*Original Count*")
print("Not Null Values")
print(df.Race.notnull().sum())
print(df.groupby(['Race'])['Name'].count())
df['Race'] = np.where(df['Race']=='Dog',None, df['Race'])
df['Race'] = df['Race'].str.upper()
print("")
print("*Cleaned Count*")
print("Not Null Values")
print(df.Race.notnull().sum())
print(df.groupby(['Race'])['Name'].count())

# clean Gender
print("*** Gender ***")
print("*Original Count*")
print("Not Null Values")
print(df.Gender.notnull().sum())
print(df.groupby(['Gender'])['Name'].count())
print("")
df['Gender'] = df['Gender'].str.strip()
df['Gender'] = df['Gender'].str.upper()
#df['Gender'] = df['Gender'].str.upper().str.strip()
print("*Cleaned Count*")
print("Not Null Values")
print(df.Gender.notnull().sum())
print(df.groupby(['Gender'])['Name'].count())

# cleaning Smokes
print("*** Smokes ***")
print("*Original Count*")
print("Not Null Values")
print(df.Smokes.notnull().sum())
print(df.groupby(['Smokes'])['Name'].count())
print(df['Smokes'].dtypes)
print("")
df['Smokes'] = np.where(df['Smokes'] == 'No', False, df['Smokes'])
df['Smokes'] = np.where(df['Smokes'] == 'Yes', True, df['Smokes'])
df['Smokes'] = np.where(df['Smokes'] == 'FALSE', False, df['Smokes'])
df['Smokes'] = np.where(df['Smokes'] == 'TRUE', True, df['Smokes'])
print(df['Smokes'].dtypes)
df['Smokes'] = df['Smokes'].astype(bool)
print(df['Smokes'].dtypes)
print("*Cleaned Count*")
print("Not Null Values")
print(df.Smokes.notnull().sum())
print(df.groupby(['Smokes'])['Name'].count())
print("Data Type: ")
print(df['Smokes'].dtypes)


# check HeightInCms
print("*** HeightInCms ***")
print("*Original Count*")
print("Null Values")
print(df.HeightInCms.isnull().sum())
print("Outlier Count")
print(utils.colOutCount(df.HeightInCms))
print("Outlier Value")
print(utils.colOutValues(df.HeightInCms))
print("HeightInCms Is < 160")
print(df.HeightInCms[df.HeightInCms<160].count())
print("HeightInCms Is >= 180")
print(df.HeightInCms[df.HeightInCms>180].count())
print("")
# clean
df['HeightInCms'] = np.where(df['HeightInCms'] < 160, 160, df['HeightInCms'])
df['HeightInCms'] = np.where(df['HeightInCms'] > 180, 180, df['HeightInCms'])
# cleaned
print("*Cleaned Count*")
print("Null Values")
print(df.HeightInCms.isnull().sum())
print("Outlier Count")
print(utils.colOutCount(df.HeightInCms))
print("Outlier Value")
print(utils.colOutValues(df.HeightInCms))
print("HeightInCms Is < 160")
print(df.HeightInCms[df.HeightInCms<160].count())
print("HeightInCms Is >= 180")
print(df.HeightInCms[df.HeightInCms>180].count())
print("")


# check WeightInKgs
print("*** WeightInKgs ***")
print("*Original Count*")
print("Null Values")
print(df.WeightInKgs.isnull().sum())
print("Outlier Count")
print(utils.colOutCount(df.WeightInKgs))
print("Outlier Value")
print(utils.colOutValues(df.WeightInKgs))
print("WeightInKgs Is < 65")
print(df.WeightInKgs[df.WeightInKgs<65].count())
print("WeightInKgs Is > 85")
print(df.WeightInKgs[df.WeightInKgs>85].count())
print("")
# clean
df['WeightInKgs'] = np.where(df['WeightInKgs'] < 65, 65, df['WeightInKgs'])
df['WeightInKgs'] = np.where(df['WeightInKgs'] > 85, 85, df['WeightInKgs'])
# cleaned
print("*Cleaned Count*")
print("Null Values")
print(df.WeightInKgs.isnull().sum())
print("Outlier Count")
print(utils.colOutCount(df.WeightInKgs))
print("Outlier Value")
print(utils.colOutValues(df.WeightInKgs))
print("WeightInKgs Is < 65")
print(df.WeightInKgs[df.WeightInKgs<65].count())
print("WeightInKgs Is > 85")
print(df.WeightInKgs[df.WeightInKgs>85].count())
print("")


# clean State
print("*** State ***")
print("*Original Count*")
print("Not Null Values")
print(df.State.notnull().sum())
print(df.groupby(['State'])['Name'].count())
print("")
df['State'] = np.where(df['State'] == 'Georgia,xxx', 'Georgia', df['State'])
df['State'] = np.where(df['State'] == 'Illlinois', 'Illinois', df['State'])
df['State'] = np.where(df['State'] == 'WrongSpelling', 'RightSpelling', df['State'])
df['State'] = df['State'].str.upper()
print("*Cleaned Count*")
print("Not Null Values")
print(df.State.notnull().sum())
print(df.groupby(['State'])['Name'].count())

# clean Pet
print("*** Pet ***")
print("*Original Count*")
print("Not Null Values")
print(df.Pet.notnull().sum())
print(df.groupby(['Pet'])['Name'].count())
print("")
df['Pet'] = df['Pet'].str.upper()
print("*Cleaned Count*")
print("Not Null Values")
print(df.Pet.notnull().sum())
print(df.groupby(['Pet'])['Name'].count())

# clean health
print("*** Health ***")
print("*Original Count*")
print(df.Health.notnull().sum())
print(df.groupby(['Health'])['Name'].count())
print("Data Type")
print(df['Health'].dtypes)
df['Health'] = df['Health'].map(
    {1: "Good Health", 2: "Normal", 3:"Poor Health", 99:"Unknown"})
df['Health'] = np.where(df['Health'] == 'Unknown', None, df['Health'])
print("*Cleaned Count*")
print("Not Null Values")
print(df.Health.notnull().sum())
print(df.groupby(['Health'])['Name'].count())
print("")

#reading data file
#df = pd.read_csv("./data/patient-data.csv")

# clean InsStatus
print("*** InsStatus ***")
print("*Original Count*")
print("Not Null Values")
print(df.InsStatus.notnull().sum())
print(df.groupby(['InsStatus'])['Name'].count())
print("")
# use map
df['InsStatus'] = df['InsStatus'].map({"A": 1, "B": 2, "C": 3})
print("Data Type: ")
print(df['InsStatus'].dtypes)
#df['InsStatus'] = df['InsStatus'].astype(int)
print("*Cleaned Count*")
print("Not Null Values")
print(df.InsStatus.notnull().sum())
print(df.groupby(['InsStatus'])['Name'].count())
print("")

# compute BMIValue
print("*** BMIValue ***")
#df = df.assign(BMIValue=df['WeightInKgs']/(df['HeightInCms']/100)**2)
df['BMIValue'] = df['WeightInKgs'] / ( (df['HeightInCms']/100)**2 )
print("*Cleaned Count*")
print("Not Null Values")
print(df.BMIValue.notnull().sum())
print(df['BMIValue'].head())
print("")

# compute BMILabel
print("*** BMILabel ***")
df['BMILabel'] = ""
df['BMILabel'] = np.where(df['BMIValue']  < 18.5 , 'Underweight', 
                    np.where((df['BMIValue'] < 25 ) , 'Normal', 
                    np.where((df['BMIValue'] < 30) , 'Overweight', 
                    "Obese")))
print("*Cleaned Count*")
print("Not Null Values")
print(df.BMILabel.notnull().sum())
print(df.groupby(['BMILabel'])['Name'].count())
print(df['BMILabel'].head())
print("")

#write data file
df.to_csv("./data/patient-data-cleaned.csv", index=False)
