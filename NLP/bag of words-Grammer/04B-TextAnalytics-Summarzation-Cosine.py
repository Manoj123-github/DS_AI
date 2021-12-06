
# https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70
# https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/
# https://medium.com/analytics-vidhya/sentence-extraction-using-textrank-algorithm-7f5c8fd568cd

#import nltk
from nltk.tokenize import word_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
 
# file-input.py
#f = open('./data/un-profile.txt','r')
f = open('./machine-learning.txt','r')
lstAllLines = f.readlines()
f.close()

# print file text
for strLine in lstAllLines:
    print(strLine)
print(len(lstAllLines))
print(type(lstAllLines))

# # make into sentenaces
# # split on . for all lines 
# lstAllSents = []
# for strLine in lstAllLines:
#     lstSent = strLine.split(".")
#     lstAllSents.extend(lstSent)

# print sentences
# print file text
lstAllSents = lstAllLines.copy()
for strSent in lstAllSents:
    print(strSent)
print(len(lstAllSents))
print(type(lstAllSents))

# stop word
import nltk.corpus
lstStopWords = nltk.corpus.stopwords.words('english')

# bad words / profanity
# original found at http://en.wiktionary.org/wiki/Category:English_swear_words
lstBadWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker","cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit","shitass","whore"]

# similarity matrix init
similarity_matrix = np.zeros((len(lstAllSents), len(lstAllSents)))
print(similarity_matrix)
print(type(similarity_matrix))
print(similarity_matrix.shape)

# similarity matrix process
intListLength = len(lstAllSents)
print(len(lstAllSents))
for idx1 in range(intListLength):
    for idx2 in range(intListLength):

        print()
        print("Idx 1:",idx1)
        print("Idx 2:",idx2)

        strSent1 = lstAllSents[idx1]
        strSent2 = lstAllSents[idx2]
        print("Sent1:",strSent1)
        print("Sent2:",strSent2)

        # split into words
        lstWordsSent1 = word_tokenize(strSent1)
        lstWordsSent2 = word_tokenize(strSent2)
        #print(lstWordsSent1)
        #print(lstWordsSent2)

        # retain alphabetic words: alpha_only
        lstWordsSent1 = [w for w in lstWordsSent1 if w.isalpha()]
        lstWordsSent2 = [w for w in lstWordsSent2 if w.isalpha()]
        #print(lstWordsSent1)
        #print(lstWordsSent2)

        # lower
        lstWordsSent1 = [w.lower() for w in lstWordsSent1]
        lstWordsSent2 = [w.lower() for w in lstWordsSent2]
        #print(lstWordsSent1)
        #print(lstWordsSent2)
    
        # remove stop words        
        lstWordsSent1 = [w for w in lstWordsSent1 if w not in lstStopWords]
        lstWordsSent2 = [w for w in lstWordsSent2 if w not in lstStopWords]
        #print(lstWordsSent1)
        #print(lstWordsSent2)

        # remove all bad words
        lstWordsSent1 = [w for w in lstWordsSent1 if w not in lstBadWords]
        lstWordsSent2 = [w for w in lstWordsSent2 if w not in lstBadWords]
        #print(lstWordsSent1)
        #print(lstWordsSent2)

        # combined words in sentence ... set gives unique words
        lstWordsInSents = list(set(lstWordsSent1 + lstWordsSent2))
        #print(lstWordsInSents)

        # empty vectore to store word count
        vecWordCount1 = np.zeros(len(lstWordsInSents))
        vecWordCount2 = np.zeros(len(lstWordsInSents))
        #print(vecWordCount1)
        #print(vecWordCount2)

        # build word count vector for the first sentence
        for w in lstWordsSent1:
            vecWordCount1[lstWordsInSents.index(w)] += 1
     
        # build word count vector for the second sentence
        for w in lstWordsSent2:
            vecWordCount2[lstWordsInSents.index(w)] += 1

        #print(vecWordCount1)
        #print(vecWordCount2)
        #print(type(vecWordCount1))
        #print(type(vecWordCount2))
        #print(vecWordCount1.shape)
        #print(vecWordCount2.shape)

        # cosine distance
        similarity_matrix[idx1][idx2] = 1 - cosine_distance(vecWordCount1, vecWordCount2)
        print(similarity_matrix[idx1][idx2])

        #time.sleep(2)

#print(similarity_matrix)
#print(similarity_matrix.shape)
#print(similarity_matrix.shape[0])
#print(similarity_matrix.shape[1])
#print(type(similarity_matrix))

#for idx1 in range(intListLength):
#    for idx2 in range(intListLength):
#        print(similarity_matrix[idx1][idx2])

#print(similarity_matrix[10][10])
#print(similarity_matrix[3][4])
#print(similarity_matrix[4][3])

#from scipy.stats import rankdata
#lstRankdIndex = rankdata(similarity_matrix, method='ordinal')
#print(intListLength)
#print(type(lstRankdIndex))
#print(lstRankdIndex.shape)
 
#for i in range(0,9):
#    print(lstRankdIndex[i])
#    print(lstAllSents[lstRankdIndex[i]])

# ranking algo
# constants
damping  = 0.85     # damping coefficient, usually is .85
min_diff = 1e-5     # convergence threshold
steps    = 100      # iteration steps

vecSentsRanks = np.array([1] * len(similarity_matrix))

# converts similarity matrix [x,x] to rank of sentences [x]
vecPrevSentsRanks = 0
for epoch in range(steps):
    vecSentsRanks = (1 - damping) + ( damping * np.matmul(similarity_matrix, vecSentsRanks) )
    if abs(vecPrevSentsRanks - sum(vecSentsRanks)) < min_diff:
        break
    else:
        vecPrevSentsRanks = sum(vecSentsRanks)
print(vecSentsRanks)
print(type(vecSentsRanks))
print(vecSentsRanks.shape)

# convert vector to list
vecSentsRanks = np.argsort(vecSentsRanks)
vecSentsRanks = list(vecSentsRanks)
vecSentsRanks.reverse()
print(vecSentsRanks)
print(type(vecSentsRanks))
print(len(vecSentsRanks))

# get top sentence
top_n = 5
lstTopSents = []
for index in range(top_n):
    thisSent = lstAllSents[vecSentsRanks[index]]
    lstTopSents.append(thisSent)

# print topn sentences
for strTopSent in lstTopSents:
    print(strTopSent)

