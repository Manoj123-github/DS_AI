

#import nltk
#nltk.download('all')

# imports
import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud

# file-input.py
print('\n*** Read File ***')
f = open('./un-profile.txt','r')
strAllTexts = f.read()
f.close()
print('Done ...')

# print file text
print('\n*** File Text ***')
# file text
print(strAllTexts)
# object type
print(type(strAllTexts))

# split into words
from nltk.tokenize import word_tokenize
print('\n*** Split Text To Words ***')
lstAllWords = word_tokenize(strAllTexts)
# print file text
print(lstAllWords)
# print object type
print(type(lstAllWords))

# convert the tokens into lowercase: lower_tokens
print('\n*** Convert To Lower Case ***')
lstAllWords = [t.lower() for t in lstAllWords]
print(lstAllWords)

# retain alphabetic words: alpha_only
print('\n*** Remove Punctuations & Digits ***')
import string
lstAllWords = [t.translate(str.maketrans('','','01234567890')) for t in lstAllWords]
lstAllWords = [t.translate(str.maketrans('','',string.punctuation)) for t in lstAllWords]
print(lstAllWords)

# remove all stop words
# original found at http://en.wikipedia.org/wiki/Stop_words
print('\n*** Remove Stop Words ***')
import nltk.corpus
lstStopWords = nltk.corpus.stopwords.words('english')
lstAllWords = [t for t in lstAllWords if t not in lstStopWords]
print(lstAllWords)

# remove all bad words / pofanities ...
# original found at http://en.wiktionary.org/wiki/Category:English_swear_words
print('\n*** Remove Profane Words ***')
lstProfWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker","cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit","shitass","whore"]
lstAllWords = [t for t in lstAllWords if t not in lstProfWords]
print(lstAllWords)

# remove application specific words
print('\n*** Remove App Specific Words ***')
lstSpecWords = ['rt','via','http','https','mailto']
lstAllWords = [t for t in lstAllWords if t not in lstSpecWords]
print(lstAllWords)

# retain words with len > 3
print('\n*** Remove Short Words ***')
lstAllWords = [t for t in lstAllWords if len(t)>3]
print(lstAllWords)

# create a Counter with the lowercase tokens: bag of words - word freq count
print('\n*** Word Freq Count ***')
from collections import Counter
dctWordCount = Counter(lstAllWords)
print(dctWordCount)

# print the 10 most common tokens
print('\n*** Word Freq Count - Top 10 ***')
print(dctWordCount.most_common(10))

# import WordNetLemmatizer
# https://en.wikipedia.org/wiki/Stemming
# https://en.wikipedia.org/wiki/Lemmatisation
# https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/
print('\n*** Stemming & Lemmatization ***')
from nltk.stem import WordNetLemmatizer
# instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
# Lemmatize all tokens into a new list: lemmatized
lstAllWords = [wordnet_lemmatizer.lemmatize(t) for t in lstAllWords]
print(lstAllWords)

# create a Counter with the lowercase tokens: bag of words - word freq count
print('\n*** Word Freq Count ***')
dctWordCount = Counter(lstAllWords)
print(dctWordCount)
print(type(dctWordCount))

# print the 10 most common tokens
print('\n*** Word Freq Count - Top 10 ***')
print(dctWordCount.most_common(10))
     
# conver dict to df
print('\n*** Convert To Dataframe ***')
dfWordCount  = pd.DataFrame.from_dict(dctWordCount, orient='index').reset_index()
dfWordCount.columns = ['Word','Freq']
print(dfWordCount.head(10))

# sort
print('\n*** Word Freq Count - Sorted ***')
dfWordCount = dfWordCount.sort_values('Freq',ascending=False)
print(dfWordCount.head(10))

# plot freq
# horizontal bar
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
