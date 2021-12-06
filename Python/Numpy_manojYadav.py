  # -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:53:33 2020

@author: Manoj Yadav
"""

# 01  Write a Python program to print non zero elements.
import numpy as np
x = np.array([[10, 0, 20], [15, 38, 0]])
print(x)
print(" non zero elements in the array:")
print(np.count_nonzero(x))


# Write a Python program to test a given array element-wise for finiteness (check for INF infinity or NaN not a number).
import numpy as np
x = np.array([1, 0, np.nan, np.inf])
print(x)
print(" array element-wise for finiteness :")
print(np.isfinite(x))

# 03 Python program to print positive Numbers in a List 

list1 = [11, 45, -9, -5, -58, 33] 
num = 0
 
while(num < len(list1)): 
	
	if list1[num] >= 0: 
		print(list1[num]) 

	num += 1
	
# 04 Write a Python program to print even numbers
list1 = [17,58,51,23,6,20,22]
for num in list1:
   if num % 2 == 0:
      print(num, end = " ")
      
#05 Write a Python program to test whether two arrays have same number of elements.
import numpy as np
x = np.array([11,7,69,28,85])
y = np.array([25,68,28,11,85])
print(x)
print(y)
print("Comparison element :-")
print(np.equal(x, y))
#06 Write a Python program to create an element-wise comparison (greater, greater_equal, less and less_equal) of two given arrays.

import numpy as np
x = np.array([11,7,69,28,85])
y = np.array([25,68,28,11,85])
print(x)
print(y)
print("Comparison element :-")
print(np.greater(x, y))
print(np.greater_equal(x, y))
print(np.less(x, y))
print(np.less_equal(x, y))

# 06 Write a Python program to create an element-wise comparison (equal, +/-1) of two given arrays.
import numpy as np
x = np.array([11,7,69,-28,85])
y = np.array([25,-68,28,11,85])
print(x)
print(y)
print("Comparison element :-")
print(np.equal(x,y))
num = 0
while(num < len(x)): 
	
	if x[num] >= 0: 
		print([num]) 
	num += 1
    

#07.Write a Python program to create an array of 10 zeros,10 ones, 10 fives
import numpy as np
array=np.zeros(10)
print("An array of 10 zeros:")
print(array)
array=np.ones(10)
print("An array of 10 ones:")
print(array)
array=np.ones(10)*5
print("An array of 10 fives:")
print(array)

#08. Write a Python program to create an array of the integers from 30 to 70.
import numpy as np
array=np.arange(30,71)
print("Array of the integers from 30 to70")
print(array)

# 09 Write a Python program to create an array of all the even integers from 30 to 70.
import numpy as np
array=np.arange(30,71,2)
print("Array of all the even integers from 30 to 70")
print(array) 

#10 Dot Product of the two arrays of your choice
import numpy as np
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
np.dot(a, b)