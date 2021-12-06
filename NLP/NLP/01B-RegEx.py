

# imports
import re

# file-input.py
f = open('./data/un-profile.txt','r')
strText = f.read()
f.close()

# print text from file
print(strText)

# print object type
print(type(strText))

# use re to replace / substitue
strText=re.sub(r'United','Divided',strText)
strText=re.sub(r'united','divided',strText)
strText=re.sub(r'UNITED','DIVIDED',strText)
print(strText)

# file-output.py
f = open('./data/diviednations.txt','w')
f.write(strText)
f.close()