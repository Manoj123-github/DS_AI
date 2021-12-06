

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
f = open('./data/tweets-small.txt','r')
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

# import
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import text2emotion as te

# classifier
def nltk_sentiment(sentence):
    nltk_sentiment = SentimentIntensityAnalyzer()
    sent_score = nltk_sentiment.polarity_scores(sentence)
    #print(sent_score['pos'])
    #print(sent_score['neg'])
    #print(sent_score['neu'])
    return sent_score

# classifier TextBlob
def blob_sentiment(sentence):
    sentence = TextBlob(sentence)
    sent_score = sentence.sentiment
    #print(type(sent_score))
    #print(sent_score[0])
    return sent_score
   
# classifier emotion
def emotion_sentiment(sentence):
    sent_score = te.get_emotion(sentence)
    #print(type(sent_score))
    #print(sent_score[0])
    return sent_score

# test 1
print("\n*** Test 1 - Positive Polarity ***")
txtOneLine = "Today I am very happy"
print(txtOneLine)
# using nltk
nltkResults = nltk_sentiment(txtOneLine)
print(nltkResults)
# using blob
blobResults = blob_sentiment(txtOneLine)
print(blobResults)
# emotions
emotionResults = emotion_sentiment(txtOneLine)
print(emotionResults)

# test 2
print("\n*** Test 2 - Negative Polarity ***")
txtOneLine = "Today is a bad day"
print(txtOneLine)
# using nltk
nltkResults = nltk_sentiment(txtOneLine)
print(nltkResults)
# using blob
blobResults = blob_sentiment(txtOneLine)
print(blobResults)
# emotions
emotionResults = emotion_sentiment(txtOneLine)
print(emotionResults)

# test 3
print("\n*** Test 2 - Neutral Polarity ***")
txtOneLine = "The board is clean"
print(txtOneLine)
# using nltk
nltkResults = nltk_sentiment(txtOneLine)
print(nltkResults)
# using blob
blobResults = blob_sentiment(txtOneLine)
print(blobResults)
# emotions
emotionResults = emotion_sentiment(txtOneLine)
print(emotionResults)

# print lines
print("\n*** Lines To Classify ***")
for vLine in lstLines:
    print(vLine)

# call clasiifier on entire list
print("\n*** Sentiment Classify ***")
# using nltk
nltkResults = [nltk_sentiment(t) for t in lstLines]
#print(nltkResults)
# using blob
blobResults = [blob_sentiment(t) for t in lstLines]
#print(blobResults)
# using blob
emotionResults = [emotion_sentiment(t) for t in lstLines]
#print(emotionResults)
print("Done ...")

# check
print("\n*** Print Sample ***")
print(lstLines[0])
print(nltkResults[0])
print(blobResults[0])
print(emotionResults[0])

# find result
def getNltkResult(pos, neu, neg):
    if (pos > neu and pos > neg):
        return ("Positive")
    elif (neg > neu and neg > pos):
        return ("Negative")
    else:
        return('Neutral')

# find result
def getBlobPolarityResult(score):
    if (score > 0.5 ):
        return ("Positive")
    elif (score < -0.5):
        return ("Negative")
    else:
        return('Neutral')

# find result
def getBlobSubjectivityResult(score):
    if (score < 0.2 ):
        return ("Very Objective")
    elif (score < 0.4):
        return ("Objective")
    elif (score < 0.6):
        return('Neutral')
    elif (score < 0.8):
        return ("Subjective")
    else:
        return ("Very Subjective")

# find result
def getEmotionResult(happy, angry, surprise, sad, fear):
    lstEmotionLabel = ['happy', 'angry', 'surprise', 'sad', 'fear']
    lstEmotionValue = [happy, angry, surprise, sad, fear]
    if max(lstEmotionValue) == 0:
        return "Neutral"
    maxIndx = lstEmotionValue.index(max(lstEmotionValue))    
    return (lstEmotionLabel[maxIndx])

getEmotionResult(5,4,3,2,1)
getEmotionResult(1,2,3,4,5)
getEmotionResult(5,4,5,2,1)
getEmotionResult(0,0,0,0,0)

# create dataframe
df = pd.DataFrame(lstLines, columns=['Lines'])
# dataframe
print("\n*** Update Dataframe - Nltk Sentiments ***")
df['Pos']=[t['pos'] for t in nltkResults]
df['Neu']=[t['neu'] for t in nltkResults]
df['Neg']=[t['neg'] for t in nltkResults]
df['NltkResult']= [getNltkResult(t['pos'],t['neu'],t['neg']) for t in nltkResults]
print("Done ...")

# dataframe
print("\n*** Update Dataframe - TextBlob Sentiments ***")
df['BlobPolarity'] = [t[0] for t in blobResults]
df['PolarityResult']= [getBlobPolarityResult(t[0]) for t in blobResults]
# create dataframe
df['BlobSubjectivity'] = [t[1] for t in blobResults]
df['SubjectivityResult']= [getBlobSubjectivityResult(t[1]) for t in blobResults]
print("Done ...")

# dataframe
print("\n*** Update Dataframe - Emotions ***")
df['Happy']=[t['Happy'] for t in emotionResults]
df['Angry']=[t['Angry'] for t in emotionResults]
df['Surprise']=[t['Surprise'] for t in emotionResults]
df['Sad']=[t['Sad'] for t in emotionResults]
df['Fear']=[t['Fear'] for t in emotionResults]
df['emotionResult']= [getEmotionResult(t['Happy'],t['Angry'],t['Surprise'],t['Sad'],t['Fear']) for t in emotionResults]
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
print(df.groupby('emotionResult').size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df['emotionResult'],label="Count")
plt.title('Emotions')
plt.show()
