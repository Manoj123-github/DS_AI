
# Containers
# Python includes several built-in container types: dictionaries, sets,
# and tuples.

# Dictionaries
# A dictionary stores (key, value) pairs, similar to a Map in Java or an 
# object in Javascript. 
# You can use it like this:
d = {'cat':'cute', 'dog':'furry'}   # Create a new dictionary with some data
print(d['cat'])                     # Get an entry from a dictionary; prints "cute"
print('cat' in d)                   # Check if a dictionary has a given key; prints "True"
print('cute' in d)                  # Check if a dictionary has a given key; prints "False"
print('junk' in d)                  # Check if a dictionary has a given key; prints "False"

d['fish'] = 'wet'                   # Set an entry in a dictionary
print(d)
print(d['fish'])                    # Prints "wet"

print(d['monkey'])                  # KeyError: 'monkey' not a key of d

print(d.get('monkey', 'N/A'))       # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))         # Get an element with a default; prints "wet"

del d['fish']                       # Remove an element from a dictionary
print(d.get('fish', 'N/A'))         # "fish" is no longer a key; prints "N/A"

# make dict
d = {'cat':'cute', 'dog':'furry'}   # Create a new dictionary with some data

# access an element
print(d['cat'])                     # Get an entry from a dictionary; prints "cute"

# keys exists in dict      
print('cat' in d)                   # Check if a dictionary has a given key; prints "True"
     
# access an element
d = {'cat':'cute', 'dog':'furry'}   # Create a new dictionary with some data
print(d)                     # Get an entry from a dictionary; prints "cute"

# get a list of all the items
print(d.items())

# get a list of all the keys
print(d.keys())

# get a list of all the values
print(d.values())

# add a key,val
d['tiger']='roaring'
print(d.keys())
print(d.values())

# change an entry
d['cat'] = "not cute"
print(d.keys())
print(d.values())

# delete an entry
del d['tiger']
print(d.keys())
print(d.values())

k = list(d.keys())
print(k)
v = list(d.values())
print(v)

# make a copy
d = {'cat':'cute', 'dog':'furry'}   # Create a new dictionary with some data
x = d.copy()
print(x.keys())
print(x.values())

# remove all items
x.clear()
print(x.keys())
print(x.values())

# number of items
print(len(d))

# looping over keys
for key in d.keys(): 
    print(key)

# looping over values
for val in d.values(): 
    print(val)

#using the if statement to get the values
if "cat" in d:
    print(d['cat'])
    
if "man" not in d:
    d['man']='human'
print(d.keys())
print(d.values())

if "man" in d:
    del d['man']
print(d.keys())
print(d.values())

# Loops: 
# It is easy to iterate over the keys in a dictionary:
d = {'person': 2, 'cat': 4, 'spider': 8}
for keys in d:
    vals = d.get(keys, 'N/A')
    print('A %s has %d legs' % (keys, vals))
# Prints "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"

# If you want access to keys and their corresponding values, use the 
# iteritems method:
d = {'person': 2, 'cat': 4, 'spider': 8}
for keys,vals in d.items():
    print('A %s has %d legs' % (keys, vals))
# Prints "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"


# update
d1={'a':1,'b':2}
dnew = d1.copy()
print(dnew.keys())
print(dnew.values())

d2={'c':3,'d':4}
dnew.update(d2)
print(dnew.keys())
print(dnew.values())

d2={'c':30,'d':40}
dnew.update(d2)
print(dnew.keys())
print(dnew.values())

d3={'e':5,'f':6}
dnew.update(d3)
print(dnew.keys())
print(dnew.values())

# update
d1={'a':1,'b':2}
dnew = d1.copy()
print(dnew.keys())
print(dnew.values())

dnew.update(d1)
print(dnew.keys())
print(dnew.values())


# Sets
# A set is an unordered collection of distinct elements. As a simple example, 
# consider the following:

# create set    
animals = {'cat', 'dog'}
print(animals)

# check if exists
print('cat' in animals)             # Check if an element is in a set; prints "True"
print('fish' in animals)            # prints "False"

# add value in set
animals.add('fish')                 # Add an element to a set
print(animals)
print('fish' in animals)            # Prints "True"
animals.add('spider')               # Add an element to a set
print(animals)
print('splder' in animals)          # Prints "False"
print('spider' in animals)          # Prints "True"

# len of set ... number of elements
print(len(animals))                 # Number of elements in a set; prints "3"

# adding an element that is already in the set does nothing
animals.add('cat')                  # Adding an element that is already in the set does nothing
print(animals)
print(len(animals))                 # Prints "3"

# remove value in set
animals.remove('cat')               # Remove an element from a set
print(animals)
print(len(animals))                 # Prints "2"

# concatenate
a1 = {'cat', 'dog', 'fox'}
print(a1)
a2 = {'lion', 'tiger', 'fox'}
print(a2)
#print(a1|a2)
new = a1.copy()
new.update(a2)
print(new)
     
# Loops: 
# Iterating over a set has the same syntax as iterating over a list; 
# however since sets are unordered, you cannot make assumptions about the 
# order in which you visit the elements of the set:
animals = {'cat', 'dog', 'fish'}
for counter, animal in enumerate(animals):
    print('#%d: %s' % (counter+1, animal))
# Prints "#1: fish", "#2: dog", "#3: cat"

# Tuples
# A tuple is an (immutable) ordered list of values. 
# A tuple is in many ways similar to a list; one of the most important 
# differences is that tuples can be used as keys in dictionaries and 
# as elements of sets, while lists cannot. 

# create tupple
t = (5, 6, 7, 8, 9)                 # Create a tuple

# print tupple
print(t)                            # prints all

# print object type
print(type(t))                      # Prints "<type 'tuple'>"

# print individual elements
print(t[0])                         # Prints "5"
print(t[1:3])                    # Prints "6 7"

# length 
print(len(t))

# Concatenation 
t1 = (1, 2, 3)
t2 = (4, 5, 6)
t3 = (1, 2, 3, 4, 5, 6)
new = t1 + t2 + t3
print(new)

# Repetition
print(('Hi!') * 4)

#	membership / exists
tup = (5, 6, 7, 8, 9)                 # Create a tuple
print (3 in tup)

# iteration 
tup = (5, 6, 7, 8, 9)                 # Create a tuple
for x in tup:
   print(x)  

# compare
# understand how comparision is done
# case 1
t1 = (4, 100, 300)
t2 = (3, 500, 400)
print(t1 == t2)
print(t1 < t2)
print(t1 > t2)
# case 2
t1 = (1, 2, 3)
t2 = (4, 5, 6)
t3 = (1, 2, 3, 4, 5, 6)
t4 = (1, 2, 3)
print(t1 > t2)
print(t1 < t2)
print(t1 == t2)
print(t1 == t4)

# min / max
t = (1, 2, 3, 4, 5, 6)
print(max(t))
print(min(t))

t = ('a', 'b', 'Z', 'd', 'x')
print(max(t))
print(min(t))

# add item to tupple
# convert tupple to list
# add item to list
# convert list back to tupple 
t = (1, 2, 3, 4, 5, 6)
n = 4
l = list(t)
l.append(n)
t = tuple(l)
print(t)

# remove item from tupple
# convert tupple to list
# remove item 
# convert list back to tupple 
t = (1, 2, 3, 4, 5, 6)
l = list(t)
l.remove(4)
t = tuple(l)
print(t)

# tupple of tupples
tuples = (('hello','there'), ('these', 'are'), ('my', 'tuples!'))
print(sum(tuples, ()))
