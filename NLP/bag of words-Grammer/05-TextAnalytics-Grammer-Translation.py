

# install from anacondo prompt
#pip install -U textblob
#python -m textblob.download_corpora
# textblob uses nltk library us nltk.download('all') must have been run

# imports
from textblob import TextBlob

# define string with textblob
t = TextBlob("Python is a high-level, general-purpose programming language.")
print(type(t))


# =============================================================================
# Part-of-speech Tagging
# =============================================================================
tags = t.tags
print(tags)
print(type(tags))

# get individual words & tags
for tag in tags:
    print(tag)
    print(type(tag))
    print(tag[0])
    print(tag[1])
    print("")
    
# description of tags
# =============================================================================
# CC coordinating conjunction
# CD cardinal digit
# DT determiner
# EX existential there (like: “there is” … think of it like “there exists”)
# FW foreign word
# IN preposition/subordinating conjunction
# JJ adjective ‘big’
# JJR adjective, comparative ‘bigger’
# JJS adjective, superlative ‘biggest’
# LS list marker 1)
# MD modal could, will
# NN noun, singular ‘desk’
# NNS noun plural ‘desks’
# NNP proper noun, singular ‘Harrison’
# NNPS proper noun, plural ‘Americans’
# PDT predeterminer ‘all the kids’
# POS possessive ending parent‘s
# PRP personal pronoun I, he, she
# PRP$ possessive pronoun my, his, hers
# RB adverb very, silently,
# RBR adverb, comparative better
# RBS adverb, superlative best
# RP particle give up
# TO to go ‘to‘ the store.
# UH interjection errrrrrrrm
# VB verb, base form take
# VBD verb, past tense took
# VBG verb, gerund/present participle taking
# VBN verb, past participle taken
# VBP verb, sing. present, non-3d take
# VBZ verb, 3rd person sing. present takes
# WDT wh-determiner which
# WP wh-pronoun who, what
# WP$ possessive wh-pronoun whose
# WRB wh-abverb where, when    
# =============================================================================

# =============================================================================
# noun phrase
# =============================================================================
nouns = t.noun_phrases
print(nouns)
    
# get individual words 
for noun in nouns:
    print(noun)
    print(type(noun))
    print("")
    
# =============================================================================
# tokenization    print(noun)
    print(noun)

# =============================================================================
t = TextBlob("Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex. Oh wow! How are you? Bye.")
print(t.words)
print(t.sentences)
    
# get individual words 
words = t.words
for word in words:
    print(word)
    print(type(word))
    print("")
    
# get individual sentence
sents = t.sentences
for sent in sents:
    print(sent)
    print(type(sent))
    print("")

# =============================================================================
# singular / plural
# =============================================================================
# sing word list    
t = TextBlob("bat is to also between use space thief key work")
l = t.words
print(l.pluralize())

# does not work so we do selctively
# get individual words 
tags = t.tags
# get individual words & tags
for tag in tags:
    print(tag)
    print(tag[0])
    print(tag[1])
    if (tag[1]=='NN'):
        print(tag[0].pluralize())
    print("")    
        
# plural word list    
t = TextBlob("bats bat is to also between uses space thives keys works")
l = t.words
print(l.singularize())

#does note work so we do selctively
# get individual words 
tags = t.tags
# get individual words & tags
for tag in tags:
    print(tag)
    print(tag[0])
    print(tag[1])
    if (tag[1]=='NNS'):
        print(tag[0].singularize())
    print("")    
    
# =============================================================================
# Lemmatization
# =============================================================================
from textblob import Word
w = Word("takes")
print(w.lemmatize("v"))
w = Word("taken")
print(w.lemmatize("v"))
w = Word("took")
print(w.lemmatize("v"))
w = Word("taking")
print(w.lemmatize("v"))
w = Word("going")
print(w.lemmatize("v"))
w = Word("gone")
print(w.lemmatize("v"))
w = Word("went")
print(w.lemmatize("v"))
w = Word("books")
print(w.lemmatize("n"))
w = Word("looks")
print(w.lemmatize("n"))
w = Word("am")
print(w.lemmatize("v"))
w = Word("are")
print(w.lemmatize("v"))
w = Word("were")
print(w.lemmatize("v"))
w = Word("is")
print(w.lemmatize("v"))


# =============================================================================
# spell check
# =============================================================================
w = Word("havv")
print(w.spellcheck())        
w = w.correct()
print(w)

# plural word list    
t = TextBlob("Data science is an inter-disciplinary fild that uses scientfic methods, processes, algoriths and systems to extract knwledge and insigts from many structural and unstructured data. Data science is related to data mining and big data.")
print(t.spellcheck())        # does not work ... no such methid
print(t.correct())        

# get individual words 
words = t.words
for w in words:
    print(w.spellcheck())
    print(w.correct())        
    print("")
    
    
# =============================================================================
# Translation and Language Detection    
# =============================================================================
t = TextBlob(u'Simple is better than complex.')
print(t.detect_language())
#print(t.detect_language())
print(t.translate(to='es'))
print(t.translate(to='fr'))
print(t.translate(to='hi'))
print(t.translate(to='ja'))
print(t.translate(to='ar'))

t = TextBlob(u'How are you today?')
print(t.detect_language())
#print(t.translate(to='en'))
print(t.translate(to='es'))
print(t.translate(to='fr'))
print(t.translate(to='hi'))
print(t.translate(to='ja'))
print(t.translate(to='ar'))

t = TextBlob(u'Bon jour!!!')
print(t.detect_language())
print(t.translate(to='en'))
print(t.translate(to='es'))
#print(t.translate(to='fr'))
print(t.translate(to='hi'))
print(t.translate(to='ja'))
print(t.translate(to='ar'))

t = TextBlob(u'Comprandez amigos?')
print(t.detect_language())
print(t.translate(to='en'))
#print(t.translate(to='es'))
print(t.translate(to='fr'))
print(t.translate(to='hi'))
print(t.translate(to='ja'))
print(t.translate(to='ar'))

t = TextBlob(u'تصبح على خير')
print(t.detect_language())
print(t.translate(to='en'))

t = TextBlob(u"美丽优于丑陋")
print(t.detect_language())
print(t.translate(to='en'))

t = TextBlob(u'さようなら')
print(t.detect_language())
print(t.translate(to='en'))

t = TextBlob(u'नमस्कार')
print(t.detect_language())
print(t.translate(to='en'))


