
# install anaconda propmpt
# pip install wordcloud
# or 
# conda install -c conda-forge wordcloud

# imports
import re
import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud

# file-input.py
print('\n*** Read File ***')
f = open('./data/un-profile.txt','r')
strAllTexts = f.read()
f.close()
print('Done ...')

# print file text
print('\n*** File Text ***')
# file text
print(strAllTexts)
# object type
print(type(strAllTexts))

# print number of chars
print('\n*** Total Chars ***')
vTotChars = len(strAllTexts)
print(vTotChars)

# split text into each line ... put into a list
print('\n*** Split Text To Lines ***')
lstAllLines = strAllTexts.split('\n')
print(lstAllLines[0])
print(lstAllLines[1])

# print Lines
print('\n*** Total Lines ***')
vTotLines = len(lstAllLines)
print(vTotLines)

# split each line into a list of words
print('\n*** Split Lines To Words ***')
lstTmpWords = []
for i in range(0,len(lstAllLines)):
    #print(i)
    strLine = lstAllLines[i]
    lstWords = strLine.split(" ")
    #print(lstWords)
    lstTmpWords.append(lstWords)
print(lstTmpWords[0])
print(lstTmpWords[1])

# merge in single list
print('\n*** Merge Words Into Single List ***')
lstAllWords = []    
for lstWords in lstTmpWords:
    for strWord in lstWords:
        lstAllWords.append(strWord)
print(lstAllWords)
    
# print words
print('\n*** Total Words ***')
vTotWords = len(lstAllWords)
print(vTotWords)

# remove unwated stuff
print('\n*** Clean List ***')
for i in range(0,len(lstAllWords)):
    lstAllWords[i] = re.sub(r'[0123456789]', '', lstAllWords[i])
    lstAllWords[i] = re.sub(r'[`~@#$%^&*()_+-=<>,.:;]', '', lstAllWords[i])
    lstAllWords[i] = re.sub(r'[–]', '', lstAllWords[i])
    lstAllWords[i] = re.sub(r'[\[\]\(\)\{\}]', '', lstAllWords[i])
    lstAllWords[i] = re.sub(r'[\t\"\'\/\\]', '', lstAllWords[i])
print(lstAllWords)

# remove short words
print('\n*** Remove Short Words ***')
lstTmpWords=[]
for strWord in lstAllWords:
    if len(strWord)>3:
        # do something with item
        lstTmpWords.append(strWord)
lstAllWords = lstTmpWords
del lstTmpWords
print(lstAllWords)

# change case
print('\n*** Convert To Lower Case ***')
for i in range(0,len(lstAllWords)):
    lstAllWords[i] = str.lower(lstAllWords[i])
print(lstAllWords)

# convert to dataframe
print('\n*** Convert To Dataframe ***')
dfWordData = pd.DataFrame({'Words':lstAllWords})
print(dfWordData.head(20))

# remove all stop words
# original found at http://en.wikipedia.org/wiki/Stop_words
print('\n*** Remove Stop Words ***')
vStopWords = ["from","all","also","and","any","are","but","can","cant","cry","due","etc","few","for","get","had","has","hasnt","have","her","here","hers","herself","him","himself","his","how","inc","into","its","ltd","may","nor","not","now","off","once","one","only","onto","our","ours","out","over","own","part","per","put","see","seem","she","than","that","the","their","them","then","thence","there","these","they","this","those","though","thus","too","top","upon","very","via","was","were","what","when","which","while","who","whoever","whom","whose","why","will","with","within","without","would","yet","you","your","yours","the"]
dfWordData = dfWordData[-dfWordData['Words'].isin(vStopWords)]
print(dfWordData.head(20))

# remove all profanities ...
# original found at http://en.wiktionary.org/wiki/Category:English_swear_words
print('\n*** Remove Profane Words ***')
vProfWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker","cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit","shitass","whore"]
dfWordData = dfWordData[-dfWordData['Words'].isin(vProfWords)]
print(dfWordData.head(20))

# get freq count
print('\n*** Significant Words Freq Count ***')
dfWordCount = pd.DataFrame(dfWordData.groupby(['Words'])['Words'].count())
dfWordCount.columns = ['Freq']
dfWordCount = dfWordCount.reset_index()
dfWordCount.columns = ['Word','Freq']
print(dfWordCount.head())
print(len(dfWordCount))

# sort
print('\n*** Significant Words Freq Count - Sorted ***')
dfWordCount = dfWordCount.sort_values('Freq',ascending=False)
print(dfWordCount.head())

# plot freq horizontal bar
print('\n*** Plot Word Freq Count - Top 15 ***')
plt.figure()
df = dfWordCount[0:15]
sns.barplot(x="Freq", y="Word", data=df, color="b", orient='h')
plt.show()

# plot word cloud
# word cloud options
# https://www.datacamp.com/community/tutorials/wordcloud-python
print('\n*** Plot Word Cloud - Top 30 ***')
d = {}
for a, x in dfWordCount[0:30].values:
    d[a] = x 
print(d)
wordcloud = WordCloud(background_color="white")
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure(figsize=[8,8])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# number of numbers in a given range
def getCount(list1, l, r):
    return len(list(x for x in list1 if l <= x < r))
 
# driver code
# freq distribution    
print('\n*** Word Freq Distribution ***')
from collections import OrderedDict
dfRange = pd.DataFrame(columns=['Range','Count'])
vStep = 5
vFrom = 1
vCount = -1
while (vCount != 0):
    vTill = vFrom + vStep
    vRange = str(vFrom) + " >= x > " + str(vTill)
    vCount = getCount(dfWordCount['Freq'], vFrom, vTill)    
    dfTmp = pd.DataFrame(OrderedDict({'Range':[vRange],'Count':[vCount]}))
    dfRange = pd.concat([dfRange,dfTmp])
    vFrom = vTill
# last one
vTill = dfWordCount['Freq'].max() + 1
vRange = str(vFrom) + " >= x > " + str(vTill)
vCount = getCount(dfWordCount['Freq'], vFrom, vTill)    
dfTmp = pd.DataFrame(OrderedDict({'Range':[vRange],'Count':[vCount]}))
dfRange = pd.concat([dfRange,dfTmp])
#  
dfRange.index = range(len(dfRange.index))
print(dfRange)
print(dfRange['Count'].sum())

# histogram
print('\n*** Word Freq Dist Histogram ***')
plt.figure()
#sns.distplot(dfWordCount['Freq'], size=5, kde=False, color='b')
sns.distplot(dfWordCount['Freq'], bins=len(dfRange)+1, kde=False, color='b')
#plt.ylim(0,50)
plt.show()
