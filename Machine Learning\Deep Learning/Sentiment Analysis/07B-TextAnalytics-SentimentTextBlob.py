

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

# split scene_one into sentences: sentences
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

# sentiment - polarity classifier using TextBlob
from textblob import TextBlob
def blob_sentiment(sentence):
    sentence = TextBlob(sentence)
    sent_score = sentence.sentiment
    #print(type(sent_score))
    #print(sent_score[0])
    return sent_score

# test 1
print("\n*** Test 1 - Positive Polarity ***")
txtOneLine = "Today I am very happy"
print(txtOneLine)
# using blob
blobResults = blob_sentiment(txtOneLine)
print(blobResults)

# test 2
print("\n*** Test 2 - Negative Polarity ***")
txtOneLine = "Today is a bad day"
print(txtOneLine)
# using blob
blobResults = blob_sentiment(txtOneLine)
print(blobResults)

# test 3
print("\n*** Test 2 - Neutral Polarity ***")
txtOneLine = "The board is clean"
print(txtOneLine)
# using blob
blobResults = blob_sentiment(txtOneLine)
print(blobResults)

# print lines
print("\n*** Lines To Classify ***")
for vLine in lstLines:
    print(vLine)

# call classifier on entire list
print("\n*** Sentiment Classify ***")
blobResults = [blob_sentiment(t) for t in lstLines]
print(blobResults)

# check
print("\n*** Print Sample ***")
print(lstLines[0])
print(blobResults[0])

# find result
def getBlobPolarityResult(score):
    if (score > 0.33 ):
        return ("Positive")
    elif (score < -0.33):
        return ("Negative")
    else:
        return('Neutral')

# find result
# objective - fact
# subjective - opinion
def getBlobSubjectivityResult(score):
    if (score < 0.33 ):
        return ("Objective")
    elif (score < 0.66):
        return('Neutral')
    else:
        return ("Subjective")

# create dataframe
print("\n*** Update Dataframe ***")
df = pd.DataFrame(lstLines, columns=['Lines'])
# polarity
df['BlobPolarity'] = [t[0] for t in blobResults]
df['PolarityResult']= [getBlobPolarityResult(t[0]) for t in blobResults]
# subjectivity
df['BlobSubjectivity'] = [t[1] for t in blobResults]
df['SubjectivityResult']= [getBlobSubjectivityResult(t[1]) for t in blobResults]
print("Done ...")

# head
print("\n*** Data Head ***")
print(df.head())

# class groupby count    
print("\n*** Group Counts ***")
print(df.groupby('PolarityResult').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df['PolarityResult'],label="Count")
plt.title('TextBlob Polarity')
plt.show()

# class groupby count    
print("\n*** Group Counts ***")
print(df.groupby('SubjectivityResult').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df['SubjectivityResult'],label="Count")
plt.title('TextBlob Subjectivity')
plt.show()

# class groupby count    
print("\n*** Group Counts ***")
print(df.groupby('PolarityResult').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df['PolarityResult'],label="Count")
plt.title('TextBlob Polarity')
plt.show()

# class groupby count    
print("\n*** Group Counts ***")
print(df.groupby('SubjectivityResult').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df['SubjectivityResult'],label="Count")
plt.title('TextBlob Subjectivity')
plt.show()

# filter negative polarity
dfNeg = df[df['PolarityResult']=='Negative']