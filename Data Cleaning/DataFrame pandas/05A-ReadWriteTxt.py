

# file-input.py
f = open('./data/un-profile.txt','r')
strText = f.read()
f.close()

# print file text
print(strText)

# print object type
print(type(strText))

# print first 10 chars
print(strText[0:9])

# print number of chars
print(len(strText))

# split each line into a list
lstText = strText.split('\n')
print(lstText)

# print number of lines
print(len(lstText))

# print first line
print(lstText[0])

# print all lines 
for line in lstText:
    print("\n")
    print(line)
    
# update text
strText = strText.replace('United','Divided')
strText = strText.replace('united','divided')
strText = strText.replace('UNITED','DIVIDED')
print(strText)

# write file
f = open('./data/diviednations1.txt','a')
f.write(strText)
f.close()

# write file
f = open('./data/diviednations2.txt','a')
for line in lstText:
	f.write(line)
f.close()
