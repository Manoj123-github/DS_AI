
# imports
import re

# file-input.py
f = open('./data/un-profile.txt','r')
strText = f.read()
f.close()

# print file from text
print(strText)

# print object type
print(type(strText))

# split
lstText = re.split('\n',strText)

# print object type 
print(type(lstText))

# find
vCount = 0
for strLine in lstText:
    #print(strLine)
    #print("=====")
    result = re.findall(r'united', strLine.lower())
    #print(result)
    vCount = vCount + len(result)
print(vCount)

# replace
for strLine in lstText:
    #print(strLine)
    #print("=====")
    strLine=re.sub(r'United','Divided',strLine)
    strLine=re.sub(r'united','divided',strLine)
    strLine=re.sub(r'UNITED','DIVIDED',strLine)
    #print(strLine)

# file-output.py
f = open('./data/diviednations.txt','w')
for strLine in lstText: 
    f.write(strLine+'\n')
f.close()