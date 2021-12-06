
#import nltk
#nltk.download('all')

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


#############################################################
# gensim
# https://radimrehurek.com/gensim_3.8.3/summarization/summariser.html
#############################################################

#import gensim
from gensim.summarization.summarizer import summarize

# pass the document along with desired word count to get the summary
my_summary = summarize(strAllTexts, word_count=200)
print(my_summary)


#############################################################
# lexrank
# https://iq.opengenus.org/lexrank-text-summarization/
#############################################################

#import sumy
from sumy.summarizers.lex_rank import LexRankSummarizer

# plain text parsers since we are parsing through text
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

# parser object with AllTexts
parserObject = PlaintextParser.from_string(strAllTexts,Tokenizer("english"))
# summarizer object
summarizer = LexRankSummarizer()
# create summary
my_summary = summarizer(parserObject.document,2)
print(my_summary)


#############################################################
# luhn summary
# https://iq.opengenus.org/luhns-heuristic-method-for-text-summarization/
#############################################################

#import sumy
from sumy.summarizers.luhn import LuhnSummarizer

#Plain text parsers since we are parsing through text
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

# parser object with AllTexts
parserObject = PlaintextParser.from_string(strAllTexts,Tokenizer("english"))
# summarizer object
summarizer = LuhnSummarizer()
# create summary
my_summary = summarizer(parserObject.document,2)
print(my_summary)


#############################################################
# lsa summary
# https://iq.opengenus.org/latent-semantic-analysis-for-text-summarization/
#############################################################

#import sumy
from sumy.summarizers.lsa import LsaSummarizer

# plain text parsers since we are parsing through text
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

# parser object with AllTexts
parserObject = PlaintextParser.from_string(strAllTexts,Tokenizer("english"))
# summarizer object
summarizer = LsaSummarizer()
# create summary
my_summary = summarizer(parserObject.document,1)
print(my_summary)



