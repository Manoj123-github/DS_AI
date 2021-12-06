

import nltk
nltk.download('all')

# imports
#nil

# file-input.py
#f = open('./data/un-profile.txt','r')
f = open('./machine-learning.txt','r')
strAllTexts = f.read()
f.close()
print('Done ...')

# print file text
print('\n*** File Text ***')
# file text
print(strAllTexts)
# object type
print(type(strAllTexts))

# summary of n lines
print('\n*** n Line Summary ***')
nLineSmry = 5
print(nLineSmry)

#############################################################
# compute word freq & word weight
#############################################################

# split into words
print('\n*** Split Text To Words ***')
#import nltk
from nltk.tokenize import word_tokenize
# split 
lstAllWords = word_tokenize(strAllTexts)
# print file text
print(lstAllWords)
# print object type
print(type(lstAllWords))

# Convert the tokens into lowercase: lower_tokens
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

# remove all bad words ...
# original found at http://en.wiktionary.org/wiki/Category:English_swear_words
print('\n*** Remove Profane Words ***')
lstBadWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker","cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit","shitass","whore"]
lstAllWords = [t for t in lstAllWords if t not in lstBadWords]
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
# import Counter
from collections import Counter
dctWordCount = Counter(lstAllWords)
print(dctWordCount)
print(type(dctWordCount))

# print the 10 most common tokens
print('\n*** Word Freq Count - Top 10 ***')
print(dctWordCount.most_common(10))

# word weight = word-count / max(word-count)
# replace word count with word weight
print('\n*** Word Weight ***')
max_freq = sum(dctWordCount.values())
print(max_freq)
for word in dctWordCount.keys():
    dctWordCount[word] = (dctWordCount[word]/max_freq)
# weights of words
print(dctWordCount)

#############################################################
# create sentences / lines
#############################################################

# split scene_one into sentences: sentences
print('\n*** Split Text To Sents ***')
from nltk.tokenize import sent_tokenize
lstAllSents = sent_tokenize(strAllTexts)
# print file text
print(lstAllSents)
# print object type
print(type(lstAllSents))

# print line count
print('\n*** Sents Count ***')
print(len(lstAllSents))

# convert into lowercase
print('\n*** Convert To Lower Case ***')
lstAllSents = [t.lower() for t in lstAllSents]
print(lstAllSents)

# remove punctuations
print('\n*** Remove Punctuations & Digits ***')
import string
lstAllSents = [t.translate(str.maketrans('','','[]{}<>')) for t in lstAllSents]
lstAllSents = [t.translate(str.maketrans('','','0123456789')) for t in lstAllSents]
print(lstAllSents)

# sent score
print('\n*** Sent Score ***')
dctSentScore = {}
for Sent in lstAllSents:
    for Word in nltk.word_tokenize(Sent):
        if Word in dctWordCount.keys():
            if len(Sent.split(' ')) < 30:
                if Sent not in dctSentScore.keys():
                    dctSentScore[Sent] = dctWordCount[Word]
                else:
                    dctSentScore[Sent] += dctWordCount[Word]
print(dctSentScore)


#############################################################
# summary of the article
#############################################################
# The "dctSentScore" dictionary consists of the sentences along with their scores. 
# Now, top N sentences can be used to form the summary of the article.
# Here the heapq library has been used to pick the top 5 sentences to summarize the article
print('\n*** Best Sent Score ***')
import heapq
lstBestSents = heapq.nlargest(nLineSmry, dctSentScore, key=dctSentScore.get)
for vBestSent in lstBestSents:
    print('\n'+vBestSent)
#print(type(lstBestSents))

# final summary
print('\n*** Text Summary ***')
strTextSmmry = '. '.join(lstBestSents) 
strTextSmmry = strTextSmmry.translate(str.maketrans(' ',' ','\n'))
print(strTextSmmry)
