
# Regular Expression
# What is Regular Expression and how is it used?
# Simply put, regular expression is a sequence of character(s) mainly used to 
# find and replace patterns in a string or file. The Regular Expression are 
# supported by most of the programming languages like python, perl, R, Java 
# and many others. 

# Regular expressions use two types of characters:
# a] Meta characters: As the name suggests, these characters have a special 
#    meaning, similar to * in wild card.
# b] Literals (like a,b,1,2…)

# In Python, we have module "re" that helps with regular expressions. 
# So we need to import library re before you can use regular expressions in Python.

# The most common uses of regular expressions are:
# -- Search a string (search and match)
# -- Finding a string (findall)
# -- Break string into a sub strings (split)
# -- Replace part of a string (sub)
# Let’s look at the methods that library "re" provides to perform these tasks.

# imports
import re

# raw
print('the quick brown fox \n jumped over the lazy dog')
print(r'the quick brown fox \n jumped over the lazy dog')

# re.match(pattern, string):
# this method finds match if it occurs at start of the string. 
result = re.match(r'the', 'the quick brown fox jumped over the lazy dog')
print(result)
print(result.group(0))

# There are methods like start() and end() to know the start and end position 
# of matching pattern in the string.
result = re.match(r'the quick', 'the quick brown fox jumped over the lazy dog')
print(result)
print(result.group(0))
print(result.start())
print(result.end())

# Above you can see that start and end position of matching pattern 
# in the string and sometime it helps a lot while performing manipulation 
# with the string.

# re.search(pattern, string):
# It is similar to match() but it doesn’t restrict us to find matches at the 
# beginning of the string only. Unlike previous method, here searching for 
# pattern ‘Analytics’ will return a match.
import re
result = re.search(r'fox', 'the quick brown fox jumped over the white fox')
print(result)
print(result.group(0))
print(result.start())
print(result.end())
# Here you can see that, search() method is able to find a pattern from any 
# position of the string but it only returns the first occurrence of the 
# search pattern.
 
# re.findall (pattern, string):
# It helps to get a list of all matching patterns. It has no constraints of 
# searching from start or end. If we will use method findall to search pattern 
# in given string it will return both occurrence of the or fox. While searching 
# a string, it is recommended you to use re.findall() always, it can work like 
# re.search() and re.match() both.
import re
result = re.findall(r'fox', 'the quick brown fox jumped over the white fox')
print(result)
print(result[0])
print(result[1])
print(result[0].start())
print(result[0].end())
 
# re.split(pattern, string, [maxsplit=0]):
# This methods helps to split string by the occurrences of given pattern.
import re
result=re.split(r' ','the quick brown fox jumped over the lazy dog')
print(result)
print(type(result))

# 
for word in result:
    print(word)

result=re.split(r'o','the quick brown fox jumped over the lazy dog')
print(result)
# it has performed all the splits that can be done by pattern "o".

result=re.split(r'o','the quick brown fox jumped over the lazy dog', maxsplit=1)
print(result)
# here, you can notice that we have fixed the maxsplit to 1. And the result is, 
# it has only two values whereas first example has three values.

# two char split
result=re.split(r'ui','the quick brown fox jumped over the lazy dog')
print(result)
result=re.split(r'fox','the quick brown fox jumped over the lazy dog')
print(result)
result=re.split(r'fox','the quick brown fox jumped over the fox lazy dog')
print(result)

# re.sub(pattern, repl, string):
# It helps to search a pattern and replace with a new sub string. If the pattern 
# is not found, string is returned unchanged.
import re
result=re.sub(r'fox','cat','the quick brown fox jumped over the lazy dog')
print(result)

# re.compile(pattern, repl, string):
# We can combine a regular expression pattern into pattern objects, which can 
# be used for pattern matching. It also helps to search a pattern again without 
# rewriting it.
import re
pattern=re.compile('fox')
result1=pattern.findall('the quick brown fox jumped over the lazy fox')
print(result1)
result2=pattern.findall('the quick brown fox jumped over the lazy dog')
print(result2)

#pattern match with start and end position ==> match & search 
#pattern match without start and end position ==>findall
#search & replace >sub
#split on deliminater>split


# quick Recap of various methods:
# Till now,  we looked at various methods of regular expression using a 
# constant pattern (fixed characters). But, what if we do not have a constant 
# search pattern and we want to return specific set of characters (defined by 
# a rule) from a string?  
# This can easily be solved by defining an expression with the help of pattern 
# operators (meta  and literal characters). 

# Let’s look at the most common pattern operators.
# Regular expressions can specify patterns, not just fixed characters. 
# Here are the most commonly used operators that helps to generate an 
# expression to represent required characters in a string or file. 
# It is commonly used in web scrapping and text mining to extract required information.
# Operators	Description
# .	 Matches with any single character except newline ‘\n’.
# ?	 match 0 or 1 occurrence of the pattern to its left
# +	 1 or more occurrences of the pattern to its left
# *	 0 or more occurrences of the pattern to its left
# \w Matches with a alphanumeric character 
# \W (upper case W) matches non alphanumeric character
# \d Matches with digits [0-9] 
# \D (upper case D) matches with non-digits.
# \s Matches with a single white space character (space, newline, return, tab, form) and \S (upper case S) matches any non-white space character.
# \b boundary between word and non-word and /B is opposite of /b
# [..] Matches any single character in a square bracket and [^..] matches any single character not in square bracket
# \	 It is used for special meaning characters like \. to match a period or \+ for plus sign.
# ^ and $	 ^ and $ match the start or end of the string respectively
# {n,m}	 Matches at least n and at most m occurrences of preceding expression if we write it as {,m} then it will return at least any minimum occurrence to max m preceding expression.
# a| b	 Matches either a or b
# ( )	Groups regular expressions and returns matched text
# \t, \n, \r	 Matches tab, newline, return
# For more details, you can refer this link (https://docs.python.org/2/library/re.html).


# Problem-1  Extract each character (using "\w")
import re
result=re.findall(r'.','the quick brown fox jumped over the lazy dog')
print(result) 
# returns spaces also
result=re.findall(r'\w','the quick brown fox jumped over the lazy dog')
print(result)
# does not returns spaces

# Problem-2  Extract each word (using "*" or "+")
result=re.findall(r'\w*','the quick brown fox jumped over the lazy dog')
print(result)
# Again, it is returning space as a word because "*" returns zero or more matches 
# of pattern to its left. Now to remove spaces we will go with "+".
result=re.findall(r'\w+','the quick brown fox jumped over the lazy dog')
print(result)

# Problem-3 return the first word (using "^")
result=re.findall(r'^\w+','the quick brown fox jumped over the lazy dog')
print(result)
# If we will use "$" instead of "^", it will return the last word
result=re.findall(r'\w+$','the quick brown fox jumped over the lazy dog')
print(result)

# Problem-4  Extract consecutive two characters of each word, excluding 
# spaces (using "\w")
result=re.findall('\w\w',r'the quick brown fox jumped over the lazy dog')
print(result)

# Problem-5  Extract consecutive two characters those available at start of 
# word boundary (using "\b\w.")
result=re.findall(r'\b\w.','the quick brown fox jumped over the lazy dog')
print(result)
result=re.findall(r'\b\w..','the quick brown fox jumped over the lazy dog')
print(result)

# Problem-6 Return the domain type of given email-ids
strText = 'this is the start abc.test@gmail.com some text xyz@lentins.co.in some text test.new@maexadata.in some more text first.test@rest.biz this is end'
# To explain it in simple manner, go with a stepwise approach:
# Extract all characters after "@"
result=re.findall(r'@\w+',strText) 
print(result) 
# Above, you can see that ".com", ".in" part is not extracted. 
result=re.findall(r'@\w+.\w+',strText) 
print(result)
# full email domain not yet extracted 
result=re.findall(r'@\w+.\w+.\w+',strText) 
print(result)
# extra \w+ to be safe
result=re.findall(r'@\w+.\w+.\w+.\w+',strText) 
print(result)
# prefix of @
result=re.findall(r'\w+.@\w+.\w+.\w+.\w+',strText) 
print(result)
result=re.findall(r'\S+@\S+',strText) 
print(result)

result=re.findall(r'@\S+',strText) 
print(result) 

# Problem-7: Return date from given string
strText = 'Amit 99-99-9999 34-3456 12-05-2007, XYZ 56-4532 11-11-2011, ABC 67-8945 12-01-2009 20-20-2020'
# Here we will use "\d" to extract digit.
result=re.findall(r'\d{2}-\d{2}-\d{4}',strText)
print(result)
result=re.findall(r'[1-31]{2}-[1-12]{2}-\d{4}',strText)
print(result)
# if you want to extract only year again parenthesis "( )" will help you.
strText = '1111 Amit 34-3456 12-05-2007, XYZ 56-4532 11-11-2011, ABC 67-8945 12-01-2009'
result=re.findall(r'\d{2}-\d{2}-\d{4}',strText)
print(result)
result=re.findall(r'\d{2}-\d{2}-(\d{4})',strText)
print(result)

# Problem-8: Return all words of a string those starts with vowel
# return each word
result=re.findall(r'\w+','the quick brown fox jumped over the lazy dog')
print(result)
# return words starts with alphabets (using [])
result=re.findall(r'[aeiouAEIOU]\w+','the quick brown fox jumped over the lazy dog')
print(result)
# now use \b to indicate use word boundry
result=re.findall(r'\b[aeiouAEIOU]\w+','the quick brown fox jumped over the lazy dog')
print(result) 
# now use \b to indicate use word boundry
result=re.findall(r'\b[aeiouAEIOU]\w+','the euick brown fox jumped over the lazy dog')
print(result) 

# In similar ways, we can extract words those starts with consonant using "^" within square bracket.
result=re.findall(r'\b[^aeiouAEIOU]\w+','the quick brown fox jumped over the lazy dog')
print(result)
# Above you can see that it has returned words starting with space. To drop it 

# from output, include space in square bracket[].
result=re.findall(r'\b[^aeiouAEIOU ]\w+','the quick brown fox jumped over the lazy dog')
print(result)

# Problem 6: validate / extract a phone number (phone number must be of 10 digits 
# and starts with 7, 8 or 9) 

# extarct 
# We have a list phone numbers in a text string and here we will extract phone 
# numbers using regular expression
strText = 'this is the start 9820098200 some text 919820098200 some text 982009820 some more text 9830098300 this is end'
result=re.findall(r'[7-9]{1}[0-9]{9}',strText)
print(result)
    
# Problem 6: Validate a phone number (phone number must be of 12 digits 
# start with 91 followed by 7, 8 or 9) 
# validate
# We have a list phone numbers in list "list" and here we will validate phone 
# numbers using regular expression
# extarct 
# We have a list phone numbers in a text string and here we will extract phone 
# numbers using regular expression
strText = 'this is the start 9820098200 some text 919820098200 some text 982009820 some more text 9830098300 this is end'
result=re.findall(r'[91]{2}[7-9]{1}[0-9]{9}',strText)
print(result)

# Problem 7: Split a string with multiple delimiters
import re
line = 'asdf fjdk;afed,fjek,asdf,foo' # String has multiple delimiters (";",","," ").
result= re.split(r'[;,\s]', line)
print(result)

# We can also use method re.sub() to replace these multiple delimiters 
# with one as space " ".
import re
line = 'asdf fjdk;afed,fjek,asdf,foo'
result= re.sub(r'[;,\s]',' ', line)
print(result)
result= re.split(r' ', result)
print(result)

# We can also use method re.sub() to replace these multiple delimiters 
# with one as space " ".
import re
line = 'asdf fjdk;a!.;,,,!!!fed,fjek,asdf,foo'
result= re.sub(r'[.,;\':]','', line)
print(result)
