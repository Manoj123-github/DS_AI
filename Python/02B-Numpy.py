
# Numpy
# Numpy is the core library for scientific computing in Python. It provides a 
# high-performance multidimensional array object, and tools for working with 
# these arrays. 

# Arrays
# A numpy array is a grid of values, all of the same type, and is indexed by  
# nonnegative integers. The number of dimensions is the rank of the 
# array; the shape of an array is integers giving the size of the 
# array along each dimension.

# We can initialize numpy arrays from nested Python lists, and access elements 
# using square brackets:

# Array Properties    
# ndarray.ndim
# the number of axes (dimensions) of the array. In the Python world, the number 
# of dimensions is referred to as rank.
# ndarray.shape
# the dimensions of the array. This is number of integers indicating the size 
# of the array in each dimension. For a matrix with n rows and m columns, 
# shape will be (n,m). The length of the shape is therefore the rank, or 
# number of dimensions, ndim.
# ndarray.size
# the total number of elements of the array. This is equal to the product of 
# the elements of shape.
# ndarray.dtype
# an object describing the type of the elements in the array. One can create or 
# specify dtype’s using standard Python types. Additionally NumPy provides types 
# of its own. numpy.int32, numpy.int16, and numpy.float64 are some examples.
# ndarray.itemsize
# the size in bytes of each element of the array. For example, an array of 
# elements of type float64 has itemsize 8 (=64/8), while one of type complex32 
# has itemsize 4 (=32/8). It is equivalent to ndarray.dtype.itemsize.
   
# import   
import numpy as np

l = [1,2,3]
print(l)

# create array
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
# print
print(a)
# dim
print(a.ndim)
# shape
print(a.shape)
# type
print(type(a))
# type
print(a.dtype.name)
# item size
print(a.itemsize)
# size
print(a.size)
# size
print(len(a))

# create array
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# print
print(a)
# dim
print(a.ndim)
# shape
print(a.shape)
# type
print(type(a))
# type
print(a.dtype.name)
# item size
print(a.itemsize)
# size
print(a.size)
# size
print(len(a))

# Array Creation
a = np.array([2,3.5,4])
print(a)
print(a.dtype)
a = np.array([1.2, 3.5, 5.1])
print(a)
print(a.dtype)

# Array Creation - Error
a = np.array(1,2,3,4)    # WRONG

a = np.array([1,2,3,4])  # RIGHT
print(a)
print(a.dtype)

a = np.array([1.0,2,3,4])  # RIGHT
print(a)
print(a.dtype)

a = np.array([1,2,3,4], dtype=np.float64 )  
print(a)
print(a.dtype)

a = np.array([1,2,3,4], dtype=np.int64 )
print(a)
print(a.dtype)

a = np.array([1.1,2.2,3.3,4.4,5.5,6.6], dtype=np.int32 )  
print(a)
print(a.dtype)

# create array of zeros
a = np.zeros( (3,4) )
print(a)

# create array of ones
a = np.ones( (3,4), dtype=np.int16 )                # dtype can also be specified
print(a)
print(a.dtype)

# create array of ones
a = np.ones( (2,3,4), dtype=np.float64 )                # dtype can also be specified
print(a)
print(a.dtype)

# create array using sequences of numbers, NumPy provides a function analogous 
# to range that returns arrays instead of lists.
a = np.arange(15)
print(a)
a = np.arange( 0, 15)
print(a)
a = np.arange( 10, 30, 5 )
print(a)
a = np.arange( 0, 2, 0.3 )                 # it accepts float arguments
print(a)
a = np.arange(15).reshape(3, 5)
print(a)
a = np.arange(15).reshape(3, 6)
print(a)
a = np.arange(27).reshape(3, 3, 3)
print(a)
a = np.arange(10000)
print(a)
a = np.arange(10000).reshape(100,100)
print(a)

# arithmetic operations
a = np.array( [20, 30, 40, 50, 60] )
b = np.arange( 4 )
print(a)
print(b)

# an array with a number
# what is the return data type

# addition
c = a + 10
print("a + 10:")
print(c)

# subtract
c = a - 10
print("a - 10:")
print(c)

# multiplication
c = a * 10
print("a * 10:")
print(c)

# division
c = a / 10
print("a / 10:")
print(c)

# raise to
c = a ** 2
print("a ** 2:")
print(a)
print(c)

# an array with another array
# what is the return data type
# arithmetic operations
a = np.array( [20,30,40,50] )
b = np.arange( 4 )
print(a)
print(b)

# addition 
c = a + b
print("a + b:")
print(c)

# subtraction
c = a - b
print("a - b:")
print(c)

# multiplication
c = a * b
print("a * b:")
print(c)

# division
c = a / b
print("a / b:")
print(c)

# raise to
c = a ** b
print("a ^ b:")
print(c)

# an array with another array
# what is the return data type
# arithmetic operations
a = np.array( [20,30,40,50] )
b = np.arange( 3 )
print(a)
print(b)

# addition 
c = a + b
print("a + b:")
print(c)

# two dim array
a = np.array( [ [2,2], [2,2] ] )
b = np.array( [ [2,2], [2,2] ] )
print(a)
print(b)

# addition
c = a + b
print("a + b:")
print(c)

# subtract
c = a - b
print("a - b:")
print(c)

# multiply 
c = a * b
print("a * b:")
print(c)

# division
c = a / b
print("a / b:")
print(c)

#https://matrix.reshish.com/multiplication.php

# two dim array
a = np.array( [ [2,2], [2,2] ] )
b = np.array( [ [2,2], [2,2] ] )
print(a)
print(b)

# dot product
c = np.dot(a,b)
print("np.dot(a,b)")
print(c)

# two dim array
a = np.arange(6).reshape(2,3)
b = np.array([6,7,8,9,10,11]).reshape(3,2)
print(a)
print(b)

# multiply 
c = a * b
print("a * b:")
print(c)

# dot product
c = np.dot(a,b)
print("np.dot(a,b)")
print(c)


# two dim array
a = np.arange(10).reshape(5,2)
b = np.array([6,7,8,9,10,11]).reshape(2,3)
print(a)
print(b)

# dot product
c = np.dot(a,b)
print("np.dot(a,b)")
print(c)

# dot product
c = np.dot(b,a)
print("np.dot(b,a)")
print(c)

# in place arithmetic operations
a = np.array( [20,30,40,50] )
b = np.arange( 4 )
#b = np.array( [2,3,4,5]  )
print(a)
print(b)

# addition
a += b 
print("a += b:")
print(a)

# subtraction
a -= b 
print("a -= b:")
print(a)

# multiplication
a *= b 
print("a *= b:")
print(a)

# division
print(a)
print(b)
a /= b 
print("a /= b:")
print(a)

# division
a = np.array( [20.0,30,40,50] )
b = np.array( [2,3,4,5]  )
print(a)
print(b)
a /= b 
print("a /= b:")
print(a)

# universal functions
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)

# count
x = a.size
print(x)

# sum
x = a.sum()
print(x)

# min
x = a.min()
print(x)

# max
x = a.max()
print(x)

# max
x = a.mean()
print(x)

# universal functions
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)

# sum rows
x = a.sum(axis=0)
print(x)

# min row
x = a.min(axis=0)
print(x)

# max rows
x = a.max(axis=0)
print(x)

# universal functions
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
print(type(a))
print(a.dtype)

# sum cols
x = a.sum(axis=1)
print(x)

# min cols
x = a.min(axis=1)
print(x)

# max cols
x = a.max(axis=1)
print(x)

#====================================================

# indexing / slicing / iterating
# one-dimensional arrays can be indexed, sliced and iterated over, much like 
# lists and other Python sequences.

# print 3 rd element
a = np.arange(10, 50, 5 )
print(a)
print(a[2])
a[2] = 0
print(a[2])
print(a)

#
a = np.arange(10, 50, 5 )
print(a)
print(a[2:5])

#
a = np.arange(15).reshape(3, 5)
print(a)
print(a[2,2])
a[2,2] = 0
print(a[2,2])
print(a)
 
# print element n:m
a = np.arange(10)
print(a)
print(a[2:5])
a[2:5] = 0
print(a[2:5])
print(a)
a[2:5] = [ 10, 20, 30]
print(a[2:5])
print(a)

#
a = np.arange(15).reshape(3, 5)
print(a)
print(a[2,2:4])
a[2,2:4] = 0
print(a)
a[2,2:4] = 100
print(a)

#
a = np.arange(15).reshape(3, 5)
print(a)
print(a[2,:])
a[2,:] = 0
print(a)
a[2,:] = 100
print(a)


#
a = np.arange(15).reshape(3, 5)
print(a)
print(a[1:3,2])
a[1:3,2] = 0
print(a)
a[1:3,2] = 100
print(a)

# 
a = np.arange(15).reshape(3, 5)
print(a)
print(a[:,2])
a[:,2] = 0
print(a)
a[:,2] = 100
print(a)

# print element n:m:s
a = np.arange(20)
print(a)
print(a[2:12:2])
a[2:12:2] = 0
print(a[2:12:2])
print(a)

# append element 
a = np.arange(20)
print(a)
a = np.append(a,[100,105,110])
print(a)
a = np.append(a,np.arange(10))
print(a)

# append row
a = [[1, 2, 3], [4, 5, 6]]
print(a)
a = np.append(a, [[7, 8, 9]], axis=0)
print(a)

# append row
a = [[1, 2], [3, 4], [5, 6]]
print(a)
a = np.append(a, [[7, 8]], axis=0)
print(a)

# insert element
a = np.arange(20)
print(a)
a = np.insert(a, 3, [100,101,103], axis=0)
print(a)

# insert row
a = np.array([[1, 1], [2, 2], [3, 3]])
print(a)
a = np.insert(a, 1, [[5,6]], axis=0)
print(a)

# insert col
a = np.array([[1, 1], [2, 2], [3, 3]])
print(a)
a = np.insert(a, 1, [[5,6,7]], axis=1)
print(a)

# delete element
a = np.arange(20)
print(a)
a = np.delete(a,1)
print(a)

# delete element
a = np.arange(20)
print(a)

# delete row
a = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print(a)
print("")
a = np.delete(a, 1, axis=0)
print(a)

# delete col
a = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print(a)
print("")
a = np.delete(a, 2, axis=1)
print(a)

# loops
a = np.arange(10)
print(a)
for i in a:
    print(i*i)


a = np.arange(42).reshape(6,7)
print(a)
print(a[3:5,3:5])
print(a[4,2:6])


# indexing / slicing / iterating
# two-dimensional arrays can be indexed, sliced and iterated over

# print single element
a = np.arange(25).reshape(5,5)
print(a)
print(a[0,0])
a[0,0] = 999
print(a[0,0])
print(a)
 
# print rows or cols
a = np.arange(25).reshape(5,5)
print(a)

# each row in the second column
print(a[0:5,1])

# equivalent to the previous example
print(a[ : ,1]) 

# all elements
print(a[:,:]) 

# each column in the second and third row 
print(a[1:3, : ])                      

# when fewer indices are provided than the number of axes, 
# the missing indices are considered complete slices:
# the last row. Equivalent to b[-1,:]
print(a[-1])

print(a[:,-1])

print(a[-1,-1])

# loops
a = np.arange(10)
print(a)
for i in a:
    i=i+2
    print(i)

# loops
a = np.arange(25).reshape(5,5)
print(a)

# iterate rows
for row in a:
    print(row)

# iterate cols
# run for loop in transpose of an array
for col in a.T:
    print(col)

# if one wants to perform an operation on each element in the array, 
# one can use the flat attribute which is an iterator over all the elements of the array:
a = np.arange(25).reshape(5,5)
print(a)
for item in a.flat:
    print(item)
    
# loop in i,j
a = np.arange(20).reshape(5,4)
print(a)
print(a.shape)
print(type(a))
rows = a.shape[0]
print("Rows:")
print(rows)
cols = a.shape[1]
print("Cols:")
print(cols)
for row in range(rows):
    for col in range(cols):
        print(a[row,col])

# copy
# the copy method makes a complete copy of the array and its data.
a = np.arange(16).reshape(4,4)
b = a
c = a.copy()
print(a)
print(b)
print(c)
# b & a is same
print(b is a)
# c & a is same
print(c is a)
# b is a ... copy of the data owned by a?
print(b.base is a.base) 
# c is a ... copy of the data owned by a?
print(c.base is a.base) 

# data changes in c
a = np.arange(16).reshape(4,4)
b = a
c = a.copy()
print(a)
print(b)
print(c)
a[0,0] = 999
print(a)
print(b)
print(c)

# data changes in a
a = np.arange(16).reshape(4,4)
b = a
print(a)
print(b)
a[1,1] = 888
b[0,0] = 999

print(a)
print(b)
c = a.copy()
print(a)
print(b)
print(c)

# view or Shallow Copy
# different array objects can share the same data. The view method creates a 
# new array object that looks at the same data.
a = np.arange(16).reshape(4,4)
v = a.view()
print(a)
print(v)
# v & a is same
print(v is a)
# c is a view of the data owned by a
print(v.base is a.base) 
# data changes in both
v[0,0] = 999
print(a)
print(v)
# data changes in both
a[3,3] = 999
print(a)
print(v)
 
# simple assignments do not make the copy of array
# object. Instead, it uses the same id() of the 
# original array to access it. The id() returns a 
# universal identifier of Python object, similar 
# to the pointer in C.
# Furthermore, any changes in either gets reflected 
# in the other. For example, the changing shape of 
# one will change the shape of the other too.
a = np.arange(6) 
print('Our array is:') 
print(a)  
print('id() of a:') 
print(id(a))  
print('a is assigned to b:') 
b = a 
print(b)  
print('id() of b:') 
print(id(b))  
print('id() of a == id() of b') 
print(id(a) == id(b))
print('Change shape of b:') 
b.shape=(3,2) 
print(b)
print('Shape of a also gets changed:')
print(a)
# b & a is same
print(b is a)
# b is a view of the data owned by a
print(b.base is a.base) 

# NumPy has ndarray.view() method which is a new 
# array object that looks at the same data of the 
# original array. Unlike the earlier case, change 
# in dimensions of the new array doesn’t change 
# dimensions of the original.
# To begin with, a is 3X2 array 
a = np.arange(6).reshape(3,2) 
print('Array a:')
print(a)  
print('Create view of a:') 
b = a.view() 
print(b)  
print('id() for both the arrays are different:') 
print('id() of a:')
print(id(a))  
print('id() of b:') 
print(id(b))
# Change the shape of b. It does not change the shape of a 
b.shape = 2,3 
print('Shape of b:') 
print(b)  
print('Shape of a:') 
print(a)
# b & a is same
print(b is a)
# b is a view of the data owned by a
print(b.base is a.base) 

# The ndarray.copy() function creates a deep copy. 
# It is a complete copy of the array and its data, 
# and doesn’t share with the original array.
a = np.array([[10,10], [2,3], [4,5]]) 
print('Array a is:')
print(a)  
print('Create a deep copy of a:') 
b = a.copy() 
print('Array b is:') 
print(b)
print('Change the contents of b:') 
b[0,0] = 100 
print('Modified array b:')
print(b)  
print('a remains unchanged:') 
print(a)
# b & a is same
print(b is a)
# b is a view of the data owned by a
print(b.base is a.base) 



a = np.arange(25).reshape(5,5)
print(a)

#print item 12
print(a[2,2])

# second row
print(a[1,:])

# fourth col
print(a[:,3])

# vals 6 7 8
print(a[1,1:4])

# vals 14 19 24
print(a[2:,4])
