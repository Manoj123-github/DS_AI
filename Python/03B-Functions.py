

#==============================================================================
# Defining a Function
# You can define functions to provide the required functionality. 
# Here are simple rules to define a function in Python.
# 
# Function blocks begin with the keyword def followed by the function name and 
# parentheses ( ( ) ).
# Any input parameters or arguments should be placed within these parentheses. 
# You can also define parameters inside these parentheses.
# The first statement of a function can be an optional statement "docstring".
# The code block within every function starts with a colon (:) and is indented.
# The statement return [expression] exits a function, optionally passing back
# an expression to the caller. A return statement with no arguments is the 
# same as return None.
# 
# Syntax
# def functionname( parameters ):
#    "function_docstring"
#    function_suite
#    return [expression]                                                                                
#==============================================================================
# define printThis 
def printThis(strTest):
   "This prints a passed string into this function"
   print(type(strTest))
   print(strTest)
   return

#==============================================================================
# Calling a Function
# You can execute it by calling it from another function or 
# directly from the Python prompt. 
# Following is the example to call printThis() function −
#==============================================================================
# call printThis 
printThis("Test1")
printThis("This prints a passed string into this function")

printThis(10)
printThis(100.09)
printThis(True)

printThis("")

printThis()


#==============================================================================
# Function Arguments
# You can call a function by using the following types of formal arguments:
# - Required arguments
# - Keyword arguments
# - Default arguments
# - Variable-length arguments
#==============================================================================

#==============================================================================
# Required Arguments
# Required arguments are the arguments passed to a function in correct 
# positional order. Here, the number of arguments in the function call should 
# match exactly with the function definition.
# 
# To call the function printme(), you definitely need to pass one argument, 
# otherwise it gives a syntax error as follows −
#==============================================================================
# define printThis 
def printThis(strTest):
   "This prints a passed string into this function"
   print(strTest)
   return
# call printThis with no params
printThis('Hello World!')
printThis('')
printThis()
printThis('Hello World!',123)

#==============================================================================
# Keyword Arguments
# Keyword arguments are related to the function calls. When you use keyword 
# arguments in a function call, the caller identifies the arguments by the 
# parameter name.
# This allows you to skip arguments or place them out of order because the 
#  Python interpreter is able to use the keywords provided to match the values 
# with parameters. 
# You can also make keyword calls to the printThis() function as below
#==============================================================================
# define printThis 
def printThis(strTest):
   "This prints a passed string into this function"
   print(strTest)
   return
# call printThis with no params
printThis(strTest="My String")
printThis(strText="My String")

#==============================================================================
# Note that the order of parameters does not matter.
#==============================================================================
# define printThis 
def printInfo(name, age):
   "This prints a passed info into this function"
   print("Name: ", name)
   print("Age ", age)
   return
# Now you can call printinfo function
printInfo(age=55, name="Cyrus")
printInfo(55, "Cyrus")
printInfo("Cyrus", 55)


#==============================================================================
# Default Arguments
# A default argument is an argument that assumes a default value if a value is 
# not provided in the function call for that argument. 
#==============================================================================
# define printThis 
def printInfo(name, age=50):
   "This prints a passed info into this function"
   print("Name: ", name)
   print("Age ", age)
   return
# Now you can call printinfo function
printInfo(name="Cyrus")
printInfo("Cyrus", 50)
printInfo(name="Cyrus", age=55)
printInfo("Cyrus", age=55)
printInfo(name="Cyrus", 50)
printInfo(name="Cyrus")
printInfo("Cyrus")


#==============================================================================
# Variable-length arguments
# You may need to process a function for more arguments than you specified 
# while defining the function. These arguments are called variable-length 
# arguments and are not named in the function definition, unlike required 
# and default arguments.
# Syntax for a function with non-keyword variable arguments is this −
# def functionname([formal_args,] *var_args_tuple):
#    "function_docstring"
#    function_suite
#    return [expression]
# An asterisk (*) is placed before the variable name that holds the values 
# of all nonkeyword variable arguments. This tuple remains empty if no 
# additional arguments are specified during the function call. 
# Following is a simple example −
#==============================================================================
# function definition is here
def printInfo(*argList):
   "This prints a variable passed arguments"
   print("Length Of List:")
   print(len(argList))
   print("Output Is:")
   for var in argList:
      print(var)
   return
# call printInfo function
printInfo(10)
printInfo(70, 60, 50)
printInfo()


# function definition is here
def printInfo(argFirst, *argList):
   "This prints a variable passed arguments"
   print("Output Is:")
   print(argFirst)
   for var in argList:
      print(var)
   return
# call printInfo function
printInfo(10)
printInfo(70, 60, 50)


# function getMean
def getMean(*argList):
    count = 0
    total = 0
    try:
        for var in argList:
            total = total + var
            count = count + 1
        mean = total / count
    except:
        mean = "N/A"        
    print(mean)
    return
getMean(70, 60, 50)
getMean(1, 2, 3, 4, 5, 6, 7, 8, 9)
getMean()
getMean(70, "60", 50)
    
#==============================================================================
# The return Statement
# The statement return [expression] exits a function, optionally passing back 
# an expression to the calling statement. A return statement with no arguments 
# is the same as return None.
# all the above examples are not returning any value. You can return a value 
# from a function as follows −
#==============================================================================
# function definition is here
def sumThese(arg1, arg2):
   # add both the parameters and return them."
   intTotal = arg1 + arg2
   print("Inside the function : ", intTotal)
   return intTotal

# Now you can call sum function
intSumOfNos = sumThese(10.4, 20.5)
print(type(intSumOfNos ))
print("Outside the function : ", intSumOfNos) 
print("Outside the function : ", sumThese(10.4, 20.5)) 

#==============================================================================
# The return Statement with multiple values
# The statement return [expression] can optionally return more than one value
#==============================================================================

# function swap
def swap(argx, argy):
    return(argy, argx)
# use
x = 10
y = 20
print(x)
print(y)    
x,y = swap(x,y)
print("")
print(x)
print(y)    

#==============================================================================
# The Lambda Functions
# These functions are called lambda / anonymous because they are not declared 
# word to create these functions.
# Lambda forms can take any number of arguments but return just one value. 
# This cannot contain commands or multiple expressions.
# lambda function cannot be a direct call to print because lambda requires
# an expression
# Lambda functions have their own local namespace and cannot access variables 
# other than those in their parameter list and those in the global namespace.
# 
# Syntax
# lambda [arg1 [,arg2,.....argn]]:expression
#==============================================================================
# lambda function 
sumThese = lambda arg1, arg2: arg1 + arg2
# Now you can call sum as a function
print("Value of total : ", sumThese(arg2=10, arg1=20)) 
print("Value of total : ", sumThese(20, 20))

def sumThese(arg1, arg2):
   intTotal = arg1 + arg2
   return intTotal
# Now you can call sum as a function
print("Value of total : ", sumThese(10, 20)) 
print("Value of total : ", sumThese(20, 20))

# lambda function 
avgThese = lambda arg1, arg2: (arg1 + arg2)/2
# Now you can call sum as a function
print("Value of Average : ", avgThese(10, 20)) 
print("Value of Average : ", avgThese(20, 20))

# lambda function 
avgThese = lambda arg1=1, arg2=1: (arg1 + arg2)/2
# Now you can call sum as a function
print("Value of Average : ", avgThese(arg1=10)) 
print("Value of Average : ", avgThese(arg2=20))
avgThese(arg1=10)
avgThese(arg2=20)


#==============================================================================
# Scope of Variables
# All variables in a program may not be accessible at all locations in that 
# program. This depends on where you have declared a variable.
# The scope of a variable determines the portion of the program where you can 
# access a particular identifier. 
# There are two basic scopes of variables in Python −
# - Global variables
# - Local variables
# Global vs. Local variables
# Variables that are defined inside a function body have a local scope, and 
# those defined outside have a global scope.
# # This means that local variables can be accessed only inside the function 
# in which they are declared, whereas global variables can be accessed 
# throughout the program body by all functions. 
# When you call a function, the variables declared inside it are brought into 
# scope. 
#==============================================================================

## local variable used inside the function
# global variable
intTotal = 1000
# function definition is here
def sumThese(arg1, arg2):
   # add both the parameters and return them."
   intTotal = intTotal + arg1 + arg2
   print("Inside the function : ", intTotal)
   return intTotal
# call sum function
# now you can call sum function
retTotal = sumThese(10,20)
print("retTotal Outside the function global total : ", retTotal)
print("intTotal Outside the function global total : ", intTotal)


## local variable used inside the function
# global variable
intTotal = 0
# function definition is here
def sumThese(arg1, arg2):
   # add both the parameters and return them."
   funTotal = arg1 + arg2
   intTotal = 100
   print("Inside the function : ", intTotal)
   return funTotal
# call sum function
# Now you can call sum function
retTotal = sumThese(10,20)
print("retTotal Outside the function global total : ", retTotal)
print("intTotal Outside the function global total : ", intTotal)

## global variable used inside the function
# global variable
intTotal = 10000
# function definition is here
def sumThese(arg1, arg2):
   # use global 
   global intTotal
   # add both the parameters and return them."
   funTotal = intTotal + arg1 + arg2
   intTotal = 10
   print("Inside the function : ", intTotal)
   return funTotal
# call sum function
# Now you can call sum function
retTotal = sumThese(10,20)
print("retTotal Outside the function global total : ", retTotal)
print("intTotal Outside the function global total : ", intTotal)

#==============================================================================
# Strings are amongst the most popular types in Python. We can create them 
# simply by enclosing characters in quotes. Python treats single quotes the 
# same as double quotes. 
# Creating strings is as simple as assigning a value to a variable. For example −
#==============================================================================
var1 = 'Hello World!'
var2 = "Python Programming"
print(var1)
print(var2)

#==============================================================================
# Accessing Values in Strings
# Python does not support a character type; these are treated as strings of 
# length one, thus also considered a substring.
# To access substrings, use the square brackets for slicing along with the 
# index or indices to obtain your substring. 
#==============================================================================
var1 = 'Hello World!'
var2 = "Python Programming"
print("var1[0]: ", var1[0])
print("var2[1:5]: ", var2[1:5])

#==============================================================================
# Updating Strings
# You can "update" an existing string by (re)assigning a variable to another 
# string. The new value can be related to its previous value or to a completely 
# different string altogether. For example −
#==============================================================================
var1 = 'Hello World!'
var1 = var1[:6] + 'Python'
print("Updated String :- ", var1)
var1 = 'Hello World!'
var1 = var1[6:] + 'Python'
print("Updated String :- ", var1)

#==============================================================================
# Escape Characters
# Following table is a list of escape or non-printable characters that can be 
# represented with backslash notation.
# An escape character gets interpreted; in a single quoted as well as double quoted strings.
# 
# \e	0x1b	Escape
# \f	0x0c	Formfeed
# \n	0x0a	Newline
# \r	0x0d	Carriage return
# \t	0x09	Tab
#==============================================================================
# test
print("[ \f ]")
print("[ \n ]")
print("[ \r ]")
print("[ \t ]")

#==============================================================================
# r/R	Raw String - Suppresses actual meaning of Escape characters. 
# The syntax for raw strings is exactly the same as for normal strings with the 
# exception of the raw string operator, the letter "r," which precedes the 
# quotation marks. The "r" can be lowercase (r) or uppercase (R) and must be 
# placed immediately preceding the first quote mark.
#==============================================================================
# test
print(r"\f")
print(r"\n")
print(r"\r")
print(r"\s")
print(r"\t")

#==============================================================================
# String Special Operators
# Assume string variable a holds 'Hello' and variable b holds 'Python', then −
# +	Concatenation - Adds values on either side of the operator	
# ... a + b will give HelloPython
# *	Repetition - Creates new strings, concatenating multiple copies of 
# the same string	
# ... a*2 will give HelloHello
# []	Slice - Gives the character from the given index	
# ... a[1] will give e
# [ : ]	Range Slice - Gives the characters from the given range	
# ... a[1:4] will give ell
# in	Membership - Returns true if a character exists in the given string	
# ... H in a will give 1
# not in	Membership - Returns true if a character does not exist in the given string	
# ... M not in a will give 1
#==============================================================================
a = 'Hello' 
b = 'Python'
print(a + b)
print(a * 2)
print(a[1])
print(a[1:4])
print('h' in a)
print('M' in a)
print('M' not in a)
print('ll' in a)
print('Py' in b)

#==============================================================================
# Triple Quotes
# Python's triple quotes comes to the rescue by allowing strings to span 
# multiple lines, including verbatim NEWLINEs, TABs, and any other special 
# characters.
# 
# The syntax for triple quotes is three consecutive single or double quotes.
# 
#==============================================================================
paraString = """this is a long string that is made up of
several lines and non-printable characters such as
TAB ( \t ) and they will show up that way when displayed.
NEWLINEs within the string, whether explicitly given like
this within the brackets [ \n ], or just a NEWLINE within
the variable assignment will also show up.
"""
print(paraString)

#==============================================================================
#
#function(variable)
#
#variable.methods()
#
#==============================================================================


#==============================================================================
# center(width, fillchar)
# Returns a space-padded string with the original string centered to a 
# total of width columns.
#==============================================================================
strString = 'Hello World!'
print("["+strString+"]")
print("["+strString.center(30, ' ')+"]")
print("["+strString.center(30, '.')+"]")
print("["+strString.center(30, 'xy')+"]")

#==============================================================================
# count(str, beg= 0,end=len(string))
# Counts how many times str occurs in string or in a substring of string 
# if starting index beg and ending index end are given.
#==============================================================================
strString = 'Hello Wo rld!'
print(strString.count(' '))
print(strString.count('o '))
print(strString.count('o ',0,7))

#==============================================================================
# len(string)
# Returns the length of the string
#==============================================================================
strString = 'Hello World!'
print(len(strString))
#print(strString.len())
#print(strString.length())
#print(strString.size())

#==============================================================================
# startswith(str, beg=0,end=len(string))
# Determines if string or a substring of string (if starting index beg and 
# ending index end are given) starts with substring str; returns true if so 
# and false otherwise.
#==============================================================================
strString = 'Hello World!'
print(strString[6:12])
print(strString.startswith('H'))
print(strString.startswith('He'))
print(strString.startswith('W',6,12))

#==============================================================================
# endswith(suffix, beg=0, end=len(string))
# Determines if string or a substring of string (if starting index beg and 
# ending index end are given) ends with suffix; returns true if so and false 
# otherwise.
#==============================================================================
strString = 'Hello World!'
print(strString[6:12])
print(strString.endswith('!!'))
print(strString.endswith('!',6,12))

#==============================================================================
# find(str, beg=0 end=len(string))
# Determine if str occurs in string or in a substring of string if starting 
# index beg and ending index end are given returns index if found and -1 
# otherwise.
#==============================================================================
strString = 'Hello WorWo!'
print(strString[6:12])
print(strString.find('#'))
print(strString.find('W'))
print(strString.find('W',6,12))
print(strString.find('Wo',6,12))
print(strString.find('!',6,12))
print(strString.find('o'))

strString = 'Hello WorWo!'
intPostn = 0
while True:
    intRetValue = strString.find('o',intPostn)
    if intRetValue == -1:
        break
    print(intRetValue)
    intPostn = intRetValue + 1

#==============================================================================
# index(str, beg=0, end=len(string))
# Same as find(), but raises an exception if str not found.
#==============================================================================
strString = 'Hello World!'
print(strString[6:12])
print(strString.index('#'))
print(strString.index('W'))
print(strString.index('W',6,12))

#==============================================================================
# rfind(str, beg=0,end=len(string))
# Same as find(), but search backwards in string.
# rindex( str, beg=0, end=len(string))
# Same as index(), but search backwards in string.
#==============================================================================
#            01234567890123456789
strString = "this is this is this"
print(strString.rfind('is'))

strString = "this is really a string example .... wow!!!"
print(strString.rfind('is'))
print(strString.rfind('is', 0, 10))
print(strString.rfind('is', 10, 0))
print(strString.find('is'))
print(strString.find('is', 0, 10))
print(strString.find('is', 10, 0))

#==============================================================================
# isalnum()
# Returns true if string has at least 1 character and all characters are 
# alphanumeric and false otherwise.
#==============================================================================
strString = 'Hello World9!'
print(strString.isalnum())
strString = 'Hello World9'
print(strString.isalnum())
strString = 'HelloWorld9'
print(strString.isalnum())
strString = 'HelloWorld'
print(strString.isalnum())

#==============================================================================
# isalpha()
# Returns true if string has at least 1 character and all characters are 
# alphabetic and false otherwise.
#==============================================================================
strString = 'Hello World!'
print(strString.isalpha())
strString = 'Hello World'
print(strString.isalpha())
strString = 'HelloWorld9'
print(strString.isalpha())
strString = 'HelloWorld'
print(strString.isalpha())

#==============================================================================
# isnumeric()
# Returns true if a unicode string contains only numeric characters and .
# false otherwise.
#==============================================================================
strString = 'Hello World!'
print(strString.isnumeric())
strString = '999'
print(strString.isnumeric())
strString = '999.99'
print(strString.isnumeric())

#==============================================================================
# islower()
# Returns true if string has at least 1 cased character and all cased 
# characters are in lowercase and false otherwise.
#==============================================================================
strString = 'Hello World!'
print(strString.islower())
strString = 'hello world!'
print(strString.islower())

#==============================================================================
# isupper()
# Returns true if string has at least one cased character and all cased 
# characters are in uppercase and false otherwise.
#==============================================================================
strString = 'Hello World!'
print(strString.isupper())
strString = 'HELLO WORLD!'
print(strString.isupper())

#==============================================================================
# istitle()
# Returns true if string is properly "titlecased" and false otherwise.
#==============================================================================
strString = 'Hello World!'
print(strString.istitle())
strString = 'HELLO WORLD!'
print(strString.istitle())

#==============================================================================
# lower()
# Converts all uppercase letters in string to lowercase.
#==============================================================================
strString = 'Hello World!'
print(strString.lower())

#==============================================================================
# upper()
# Converts lowercase letters in string to uppercase.
#==============================================================================
strString = 'Hello World!'
print(strString.upper())

#==============================================================================
# title()
# Returns "titlecased" version of string, that is, all words begin with 
# uppercase and the rest are lowercase.
#==============================================================================
strString = 'hello world!'
print(strString.title())

#==============================================================================
# capitalize()
# Capitalizes first letter of string
#==============================================================================
strString = 'hello world!'
print(strString.capitalize())
strString = 'hello world! Hello cyrus. Hello myself'
print(strString.capitalize())

#==============================================================================
# swapcase()
# Inverts case for all letters in string.
#==============================================================================
strString = 'Hello World!'
print(strString.swapcase())

#==============================================================================
# ljust(width[, fillchar])
# Returns a space-padded string with the original string left-justified to a 
# total of width columns.
# rjust(width,[, fillchar])
# Returns a space-padded string with the original string right-justified 
# to a total of width columns.
#==============================================================================
strString = 'hello world!'
print(strString.ljust(30, ' '))
strString = 'hello world!'
print(strString.rjust(30, ' '))

#==============================================================================
# zfill (width)
# Returns original string leftpadded with zeros to a total of width characters; 
# intended for numbers, zfill() retains any sign given (less one zero).
#==============================================================================
strString = '9999'
print(strString.zfill(10))

#==============================================================================
# strip([chars])
# Performs both lstrip() and rstrip() on string
# lstrip()
# Removes all leading whitespace in string.
# rstrip()
# Removes all trailing whitespace of string.
#==============================================================================
strString = '            hello world!          '
print("["+strString.strip()+"]")
print("["+strString.lstrip()+"]")
print("["+strString.rstrip()+"]")

#==============================================================================
# max(str)
# Returns the max alphabetical character from the string str.
# min(str)
# Returns the min alphabetical character from the string str.
#==============================================================================
strString = 'Zhello world!'
print(max(strString))
print(min(strString))

#==============================================================================
# replace(old, new [, max])
# Replaces all occurrences of old in string with new or at most max occurrences 
# if max given.
#==============================================================================
strString = 'Hello World!'
print(strString.replace("World","Cyrus"))
print(strString.replace("Xorld","Cyrus"))

strString = 'Hello World!'
print(strString.replace("World ","Cyrus Cyrus"))
print(strString.replace("Xorld","Cyrus"))

strString = 'Hello World! Hello World'
print(strString.replace("World ","Cyrus Cyrus"))
print(strString.replace("Xorld","Cyrus"))

print(strString.replace("World","Cy"))
print(strString.replace("Xorld","Cy"))

print(strString.replace("Wo","Cyrus"))
print(strString.replace("ld","Cyrus"))


#==============================================================================
# maketrans
# The syntax of maketrans() method is:
# string.maketrans(x[, y[, z]])
# Here, y and z are optional arguments.
#==============================================================================

# x - If only one argument is supplied, it must be a dictionary.
#The dictionary should contain 1-to-1 mapping from a single character string to 
#its translation OR a unicode number (97 for 'a') to its translation.

# example - single argument (dict)
# uses 
dict = {"a": "123", "b": "456", "c": "789"}
print(dict)
strString = "abc"
strString = strString.maketrans(dict)
print(strString)

# example - single argument (dict)
dict = {97: "123", 98: "456", 99: "789"}
print(dict)
strString = "abc"
print(strString)

# y - If two arguments are passed, it must be two strings with equal length.
# Each character in the first string is a replacement to its corresponding index 
# in the second string.

# example 2 - two argumetns - translation table using two strings with maketrans()
firstString = "abc"
secondString = "def"
strString = "abc"
strString = strString.maketrans(firstString, secondString)
print(strString)
print(type(strString))

# example 2 - two argumetns - translation table using two strings with maketrans()
firstString = "abc"
secondString = "defghi"
string = "abc"
strString = strString.maketrans(firstString, secondString)
print(strString)

# z - If three arguments are passed, each character in the third argument is 
# mapped to None.

# example 3 - three argumetns - translational table with removable string with maketrans()
firstString = "abc"
secondString = "def"
thirdString = "abd"
string = "abc"
print(string.maketrans(firstString, secondString, thirdString))


#==============================================================================
# translate
# the translate() method returns a string where each character is mapped to its 
# corresponding character as per the translation table.
# return value from String translate()
#==============================================================================

import string

# example 1
# replace aeiou with 12345
inStr = "aeiou"
otStr = "12345"
# make translate dictionary
dTrns = string.maketrans(inStr, otStr)
print(dTrns)
# translate
str = "this is a sample string ... wow!!!"
print(str.translate(dTrns))

# example 2
# replace any alpha with ""
firstString = " "
secondString = " "
thirdString = "abcdefghijklmnopqrstuvwxyz"
# maketrans dict
dTranslation = string.maketrans(firstString, secondString, thirdString)
print(dTranslation)
# make input string
strString = "abcdef12345xyz"
print("Original String:", strString)
# translate string
strTranslation = strString.translate(dTranslation)
print("Translated String:", strTranslation)

# example 3
# replace any punctuation with ""
strString = "string. With, Punctuation?" # Sample string 
dTranslat = str.maketrans('', '', string.punctuation)
outString = strString.translate(dTranslat)
print(outString)

# single line
print(strString.translate(str.maketrans(" ", " ", string.punctuation)))

# string constants
#https://docs.python.org/2/library/string.html
#string.ascii_letters
#string.ascii_lowercase
#string.ascii_uppercase
#string.digits
#string.hexdigits
#string.letters
#string.lowercase
#string.octdigits
#string.punctuation
#string.printable
#string.uppercase
#string.whitespace

#==============================================================================
# split(str="", num=string.count(str))
# Splits string according to delimiter str (space if not provided) and returns 
# list of substrings; split into at most num substrings if given.
#==============================================================================
strString = "Line1-abcdef \nLine2-abc \nLine4-abcd"
print(strString.split())
print(strString.split('a'))
print(strString.split('a',1))

# import library
import random
#random.seed(101)
# for loop
for x in range(10):
    # get random number
    intNumber = random.randrange(1, 100)
    # print
    print(intNumber)
    
    


