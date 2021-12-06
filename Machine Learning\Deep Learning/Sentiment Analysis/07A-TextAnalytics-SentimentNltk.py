

#import nltk
#nltk.download('all')
#nltk.download('vader_lexicon')

# imports
import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns

##############################################################
# Read Data 
##############################################################

# file-input.py
print("\n*** Read File ***")
f = open('./tweets-small.txt','r')
strText = f.read()
f.close()
print("Done ... ")

# print file text
print("\n*** File Text ***")
print(strText)

# print object type
print("\n*** File Text Type ***")
print(type(strText))

##############################################################
# Exploratory Data Analytics
##############################################################

# print char count
print("\n*** Char Count ***")
print(len(strText))

# split string into sentences: sentences
print("\n*** Tokenize Into Lines ***")
from nltk.tokenize import sent_tokenize
lstLines = sent_tokenize(strText)
print("Done ... ")

# print file text
print("\n*** File Lines ***")
print(lstLines)

# print object type
print("\n*** File Lines Type ***")
print(type(lstLines))

# print line count
print("\n*** File Line Count ***")
print(len(lstLines))

# number of chars in each line
print("\n*** Chars Per Line ***")
lstCharLength = [len(vline) for vline in lstLines]
print(lstCharLength)

# print line 1 & chars in line 1
print("\n*** Sample Line & Char Count ***")
print(lstLines[0])
print(lstCharLength[0])

# plot a histogram of the line lengths
# histogram
print("\n*** Histogram ***")
plt.figure()
sns.distplot(lstCharLength, bins=7, kde=False, color='b')
plt.show()

##############################################################
# Data Transformation
##############################################################

# convert into lowercase
print("\n*** Lower Case ***")
lstLines = [t.lower() for t in lstLines]
print(lstLines)

# remove punctuations
print("\n*** Remove Punctuations ***")
import string
lstLines = [t.translate(str.maketrans('','',string.punctuation)) for t in lstLines]
print(lstLines)

##############################################################
# classifier 
##############################################################

# sentiment - polarity classifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def nltk_sentiment(sentence):
    nltk_sentiment = SentimentIntensityAnalyzer()
    sent_score = nltk_sentiment.polarity_scores(sentence)
    print(sent_score)
    print(sent_score['pos'])
    print(sent_score['neg'])
    print(sent_score['neu'])
    return sent_score


# test 1
print("\n*** Test 1 - Positive Polarity ***")
txtOneLine = "Today I am very happy"
print(txtOneLine)
# using nltk
nltkResults = nltk_sentiment(txtOneLine)
print(nltkResults)

# test 2
print("\n*** Test 2 - Negative Polarity ***")
txtOneLine = "Today is a bad day"
print(txtOneLine)
# using nltk
nltkResults = nltk_sentiment(txtOneLine)
print(nltkResults)

# test 3
print("\n*** Test 2 - Neutral Polarity ***")
txtOneLine = "The board is clean"
print(txtOneLine)
# using nltk
nltkResults = nltk_sentiment(txtOneLine)
print(nltkResults)

# print lines
print("\n*** Lines To Classify ***")
for vLine in lstLines:
    print(vLine)

# call classifier on entire list
print("\n*** Sentiment Classify ***")
nltkResults = [nltk_sentiment(t) for t in lstLines]
print(nltkResults)

# check
print("\n*** Print Sample ***")
print(lstLines[0])
print(nltkResults[0])

# find result
def getNltkResult(pos, neu, neg):
    if (pos > neu and pos > neg):
        return ("Positive")
    elif (neg > neu and neg > pos):
        return ("Negative")
    else:
        return('Neutral')

# create dataframe
print("\n*** Update Dataframe ***")
df = pd.DataFrame(lstLines, columns=['Lines'])
df['Pos']=[t['pos'] for t in nltkResults]
df['Neu']=[t['neu'] for t in nltkResults]
df['Neg']=[t['neg'] for t in nltkResults]
df['NltkResult']= [getNltkResult(t['pos'],t['neu'],t['neg']) for t in nltkResults]
print("Done ...")

# head
print("\n*** Data Head ***")
print(df.head())

# check class
# outcome groupby count    
print("\n*** Group Counts ***")
print(df.groupby('NltkResult').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df['NltkResult'],label="Count")
plt.title('Nltk Polarity')
plt.show()
