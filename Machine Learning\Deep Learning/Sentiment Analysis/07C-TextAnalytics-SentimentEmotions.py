
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

# sentiment - emotion classifier
import text2emotion as te
def emotion_sentiment(sentence):
    sent_score = te.get_emotion(sentence)
    #print(type(sent_score))
    #print(sent_score[0])
    return sent_score

# test 1
print("\n*** Test 1 - Happy Emotion ***")
txtOneLine = "Today I am very happy"
print(txtOneLine)
# emotions
emotionResults = emotion_sentiment(txtOneLine)
print(emotionResults)

# test 2
print("\n*** Test 2 - Sad Emotion ***")
txtOneLine = "Today I am very sad"
print(txtOneLine)
# emotions
emotionResults = emotion_sentiment(txtOneLine)
print(emotionResults)

# test 3
print("\n*** Test 3 - Neutral Emotion ***")
txtOneLine = "The board is clean"
print(txtOneLine)
# emotions
emotionResults = emotion_sentiment(txtOneLine)
print(emotionResults)

# print lines
print("\n*** Lines To Classify ***")
for vLine in lstLines:
    print(vLine)

# call clasifier on entire list
print("\n*** Emotions Classify ***")
emotionResults = [emotion_sentiment(t) for t in lstLines]
print(emotionResults)

# check
print("\n*** Print Sample ***")
print(lstLines[0])
print(emotionResults[0])

# find result
def getEmotionResult(happy, angry, surprise, sad, fear):
    lstEmotionLabel = ['happy', 'angry', 'surprise', 'sad', 'fear']
    lstEmotionValue = [happy, angry, surprise, sad, fear]
    if max(lstEmotionValue) == 0:
        return "Neutral"
    maxIndx = lstEmotionValue.index(max(lstEmotionValue))    
    return (lstEmotionLabel[maxIndx])

# create dataframe
print("\n*** Update Dataframe ***")
df = pd.DataFrame(lstLines, columns=['Lines'])
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

