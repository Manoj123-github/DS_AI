

#==============================================================================
# Python Lists
# The list is a most versatile datatype available in Python which can be 
# written as a list of comma-separated values (items) between square brackets. 
# Important thing about a list is that items in a list need not be of the same 
# type.
#==============================================================================

#==============================================================================
# Creating a list is as simple as putting different comma-separated values 
# between square brackets. 
#==============================================================================
# empty list
lstMyList = []
print(lstMyList)

# list of integers
lstMyList = [1, 2, 3]
print(lstMyList)

# list with mixed datatypes
lstMyList = [1, "Hello", 3.4]
print(lstMyList)
 
#==============================================================================
# Accessing Values in Lists
# To access values in lists, we can use the index operator [] to access an 
# item in a list. Index starts # from 0. So, a list having 5 elements will 
# have index from 0 to 4.
# 
# Trying to access an element other that this will raise an IndexError. The 
# index must be an integer. We can't use float or other types, this will 
# result into TypeError.
# 
# Nested list are accessed using nested indexing.
#==============================================================================
lstMyList = ['physics', 'chemistry', 1997, 2000]
print("lstMyList[0]: ", lstMyList[0])

lstNewList = [1, 2, 3, 4, 5, 6, 7 ]
print("lstNewList[1:5]: ", lstNewList[1:5])

# another example
lstMyList = ['p','r','o','b','e']

# Output: p
print(lstMyList[0])

# Output: o
print(lstMyList[2])

# Output: e
print(lstMyList[4])

#
print(lstMyList[8])

#
print(lstMyList[-1])
print(lstMyList[-4])
print(lstMyList[-6])
#
print(lstMyList[0:3])
print(lstMyList[0:-3])

# Error! Only integer can be used for indexing
print(lstMyList[3.5])

#==============================================================================
# Updating Lists
# You can update single or multiple elements of lists by giving the slice on 
# the left-hand side of the assignment operator; the right side of assignment 
# opertor is the value to be assigned
#==============================================================================
lstMyList = ['physics', 'chemistry', 1997, 2000]
print(lstMyList)
print("Value available at index 2 : ")
print(lstMyList[2])
lstMyList[2] = 2001
print( "New value available at index 2 : ")
print(lstMyList[2])

#==============================================================================
# Appending Lists
# You can add to elements in a list with the append() method.
# The method append() appends a passed obj into the existing list.
# This method does not return any value but updates existing list.
#==============================================================================
lstMyList = [123, 'xyz', 'zara', 'abc']
print(lstMyList)
lstMyList.append( 2009 )
print("Updated List : ", lstMyList)

lstMyList = [123, 'xyz', 'zara', 'abc']
lstMyList.append( 2009, "zzz" )
print("Updated List : ", lstMyList)


#==============================================================================
# Delete List Elements
# To remove a list element, you can use either the del statement if you know 
# exactly which element(s) you are deleting or the remove() method if you do 
# not know. For example −
#==============================================================================
lstMyList = ['physics', 'chemistry', 1997, 2000]
print(lstMyList)
del lstMyList[2]
print("After deleting value at index 2 : ")
print(lstMyList)
del lstMyList[2]
print("After deleting value at index 2 : ")
print(lstMyList)
del lstMyList[2]
print("After deleting value at index 2 : ")
print(lstMyList)

#==============================================================================
# Basic List Operations
# Lists respond to the + and * operators much like strings; they mean 
# concatenation and repetition here too, except that the result is a new list, 
# not a string.
# In fact, lists respond to all of the general sequence operations we used on 
# strings in the prior chapter.
#==============================================================================
# length
print(len([1, 2, 3]))
# concatenation
print([1, 2, 3] + [4, 5, 6])
# repetition
print(['Hi!', 'GM!'] * 4	)
# membership
print(3 in [1, 2, 3])
# membership
print(0 in [1, 2, 3])
# membership 
lstMyList = ['xyz', 'zara', 'abc']
print('xyz' in lstMyList)
print('sss' in lstMyList)
print('XYZ' in lstMyList)
# iteration
for x in [1, 2, 3]:
    print(x)
    x = 0
    print(x)
# iteration
lstMyList = ['xyz', 'zara', 'abc']
for x in lstMyList:
    print(x)

#==============================================================================
# Built-in List Functions & Methods:
# Python includes the following list functions −
#==============================================================================

# def list
lstMyList = [1, 2, 3, 4, 5, 6]
# gives the total length of the list.
print(len(lstMyList))
# returns item from the list with max value.
print(max(lstMyList))
# returns item from the list with min value.
print(min(lstMyList))
# returns item from the list with sum value.
print(sum(lstMyList))
# mean
print(sum(lstMyList)/len(lstMyList))


# define list
lstMyList = ['english', 'french', 'maths', 'Physics', 'chemistry', 'biology']
# gives the total length of the list.
print(len(lstMyList))
# returns item from the list with max value.
print(max(lstMyList))
# returns item from the list with min value.
print(min(lstMyList))


#==============================================================================
# Python includes following list methods
#==============================================================================
# appends object obj to list
# list.append(obj)
aList = [123, 'xyz', 'zara', 'abc']
aList.append( 2009 )
print("Updated List : ", aList)

# returns count of how many times obj occurs in list
# list.count(obj)
aList = [123, 'xyz', 'zara', 'abc', 123]
print("Count for 123 : ", aList.count(123))
print("Count for zara : ", aList.count('zara'))
print("Count for zzzz : ", aList.count('zzz'))

# appends the contents of seq to list
# list.extend(seq)
aList = [123, 'xyz', 'zara', 'abc', 123, 2009]
bList = [2009, 'manni']
cList = [2009, 'manni']
#aList = aList + bList
#print(aList)
aList.extend(bList)
print("Extended List : ", aList) 
aList.extend(bList,cList)
print("Extended List : ", aList) 

# returns the lowest index in list that obj appears
# list.index(obj)
aList = [123, 'abc', 'xyz', 'zara', 'abc'];
print("Before List: ", aList)
print("Index for xyz : ", aList.index('xyz'))
print("Index for zara : ", aList.index('zara')) 
print("Index for abc : ", aList.index('abc'))
print("After List: ", aList)

# inserts object obj into list at offset index
# list.insert(index, obj)
aList = [123, 'xyz', 'zara', 'abc']
print("Final List : ", aList)
aList.insert( 3, 2009)
print("Final List : ", aList)
aList.insert( 3, 2009, 2010)
print("Final List : ", aList)

# removes and returns last object or obj from list
# list.pop(obj=list[-1])
aList = [123, 'xyz', 'zara', 'abc']
print("Pre Pass 1 List : ", aList)
aObj = aList.pop()
print("Pass 1 List : ", aList)
print("Pass 1 Obj  : ", aObj)
aObj = aList.pop()
print("Pass 2 List : ", aList)
print("Pass 2 Obj  : ", aObj)

# removes object obj from list
# list.remove(obj)
aList = [123, 'xyz', 'zara', 'abc', 'zara', 123]
print("List   : ", aList)
aList.remove(123)
print("List   : ", aList)
aList.remove('zara')
print("List   : ", aList)
aList.remove('zzz')
print("List   : ", aList)

# reverses objects of list in place
# list.reverse()
lstMyList = ['english', 'french', 'maths', 'physics', 'chemistry', 'biology']
print(lstMyList)
lstMyList.reverse()
print(lstMyList)

# sorts objects of list, use compare func if given
# list.sort([func])
lstMyList = ['english', 'french', 'maths', 'physics', 'chemistry', 'biology']
lstMyList.sort()
print(lstMyList)

# sorts objects of list, use compare func if given
# list.sort([func])
lstMyList = [2,4,6,8,1,3,5,9]
print(lstMyList)
lstMyList.sort()
print(lstMyList)

lstMyList = ['english', 'french', 'maths', 'physics', 'chemistry', 'biology',2,4,6,8,1,3,5,9]
print(lstMyList)
lstMyList.sort()
print(lstMyList)

lstMyList = [2.2,4.4,6.6,8.8,1,3,5,9]
print(lstMyList)
lstMyList.sort()
print(lstMyList)

aList = [123, 'xyz', 'zara', 'abc', 'zara', 123]
print("List   : ", aList)
aList.sort()
print("List   : ", aList)

aList = [2.2,4.4,6.6,8.8,1,3,5,9]
print("List   : ", aList)
aList.sort()
print("List   : ", aList)

# Loops
# You can loop over the elements of a list like this:
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# Prints "cat", "dog", "monkey", each on its own line.

# You can loop over the elements of a list like this:
animals = ['cat', 'dog', 'monkey']
for i in range(0,len(animals)):
    print(animals[i])

# If you want access to the index of each element within the body of a loop, 
# use the built-in enumerate function:
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print(idx, animal)

# You can loop over the elements of a list like this:
animals = ['cat', 'dog', 'monkey']
for i in range(0,len(animals)):
    print(i,animals[i])

# You can loop over the elements of a list like this:
animals = ['cat', 'dog', 'monkey']
i = 0
while True:
    if i >= len(animals):
        break
    print(i, animals[i])
    i = i + 1


###===================================

# List comprehensions: 
# When programming, frequently we want to transform one type of data into 
# another. As a simple example, consider the following code that computes 
# square numbers

nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)                      # Prints [0, 1, 4, 9, 16]


# you can make this code simpler using a list comprehension:
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)                      # Prints [0, 1, 4, 9, 16]

# list comprehensions can also contain conditions:
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)                 # Prints "[0, 4, 16]"

###===================================

names = ['Cyrus Lentin', 'Vipul', 'Abbas', 'AdiL']
hours = [200, 250, 210, 220]
rates = [1000, 1050, 1000, 1200]

for i in range(3):
    print(names[i], rates[i], hours[i])


print("Sr  Name                  Hours      Rate      Salary")
print("=== ===================== ========== ========= ============")
for i in range(4):
    print('%3d %20s %10d %10d %10d' % (i+1, names[i], hours[i], rates[i], hours[i]*rates[i]))


print("Sr  Name                  Hours     Rate       Salary")
print("=== ===================== ========= ========== ============")
for i in range(4):
    print('%3d %20s %10s %10s %12.2s' % (i+1, names[i].ljust(20, ' '), hours[i], rates[i], hours[i]*rates[i]))



























