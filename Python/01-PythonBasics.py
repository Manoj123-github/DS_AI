

# hi there

# print
print("Hello World!!!")
print("Hello Cyrus!!!")

# input
name = input("Enter your name: ")
print("Hello " + name)

# integer
quantity = 3
print(quantity)
print(type(quantity))

# float
price = 2.5
print(price)
print(type(price))

# float
price = 3.0
print(price)
print(type(price))

#boolean
status=True
print(status)
print(type(status))

# string
message="Hello World!!!"
print(message)
print(type(message))

###
# operators
###
# # https://www.w3schools.com/python/python_operators.asp

# maths operations
x = 10
y = 20
z = x + y
print(z)
z = x - y
print(z)
z = x * y
print(z)
z = x / y
print(z)
z = y ** x
print(z)
z = y ** 2
print(z)
z = y % x
print(z)


z = y * x
print(z)
print(type(z))

z = y / x
print(z)
print(type(z))

x = 10
y = 20.0
z = y % x
print(z)
print(type(z))


# auto casting
quantity = 3
price = 2.9
print(price)
print(type(price))
print(quantity)
print(type(quantity))
amount = price * quantity
print(amount)
print(type(amount))

# explicit conversion - int
x = -2.1
x = int(x)
print(x)
type(x)

# explicit conversion - int
x = -2.6
x = int(x)
print(x)
type(x)

# explicit conversion - int
x = 2.6
x = int(x)
print(x)
type(x)

# explicit conversion - int
# check the effect of commenting line #price=int(price)
quantity = 3
price = 2.9
price=int(price)
print(price)
print(type(price))
print(quantity)
print(type(quantity))
amount = price * quantity
amount = int(amount )
print(amount)
print(type(amount))

# explicit conversion - float
# check the effect of commenting line #quantity=float(quantity)
print(price)
print(type(price))
quantity=float(quantity)
print(quantity)
print(type(quantity))
amount = price * quantity
print(amount)
print(type(amount))

# auto casting does not happen with string & numeric vars
item = "Book" 
print(item)
print(type(item))
label = item + " " + str(price)
print(label)
type(label)
print(item + " " + str(price))

# auto casting happens with int & logical vars
print(quantity)
print(type(quantity))
quantity=int(quantity)
result = quantity * True 
print(result)
print(type(result))

# auto casting happens with int & logical vars
print(quantity)
print(type(quantity))
quantity=int(quantity)
result = quantity + True 
print(result)
print(type(result))

# auto casting happens with str & logical vars
print(quantity)
print(type(quantity))
quantity=int(quantity)
result = item + " " + str(False) 
print(result)
print(type(result))
 
# auto casting happens with int & logical vars
print(quantity)
print(type(quantity))
quantity=int(quantity)
result = quantity * False 
print(result)
print(type(result))


# int from string
x = "123"
print(x)
print(type(x))
x = int(x)
print(x)
print(type(x))
x = float(x)
print(x)
print(type(x))

# number from string
x = "123.8"
print(x)
print(type(x))
#x = int(x)
#print(x)
#print(type(x))
x = float(x)
print(x)
print(type(x))

# number from string
x = "12.3"
print(x)
print(type(x))
# to int
#x = int(x)
#print(x)
#print(type(x))
# to float
x = float(x)
print(x)
print(type(x))

# number from string
x = "12,3"
print(x)
print(type(x))
# to int
#x = int(x)
#print(x)
#print(type(x))
# to float
x = float(x)
print(x)
print(type(x))

# number from string
x = "12x3"
print(x)
print(type(x))
# to int
x = int(x)
print(x)
print(type(x))
# to float
x = float(x)
print(x)
print(type(x))

###
# if 
###
# https://www.w3schools.com/python/python_conditions.asp

# if statement
x = 10 
if x == 10: 
    print("ooo")
    print("xxx")
else:
    print("xxx")
print('NOTA')


# if statement
x = 10
if x == 5:
    print("Equals 5")
elif x < 4:
	print('Less than 4')
elif x < 5:
	print('Less than 5')
elif x < 6: 
	print('Less than 6')
elif x <= 5:
    print('Less than or Equals 5')
elif x != 6:
    print('Not equal 6')
if x == 10: 
    print("ooo")
else:
  print('NOTA')
print("ha ha") 


# if statement
x = 5
if x == 5:
  print("Equals 5")
elif x < 5:
	print('Less than Or Equals 6')
elif x > 5:
	print('More than Or Equals 5')
else:
  print('NOTA')
print("ha ha") 

# another if
print('Before 5')
x = 5
if x == 5:
 print('Is 5		 ')
 print('Is Still 5')
 print('Third 5   ')
 print('After 5   ') 
 print('Before 6  ')
print("End ") 
if x == 6:
    print('Is 6      ')
    print('Is Still 6')
    print('Third 6   ')
    print('After 6')
print("End ") 

# one more if
x = int(input("Please enter an integer: "))
if x < 0:
   x = 0
   print('Negative changed to zero')
elif x == 0:
   print('Zero')
elif x == 1:
   print('Single')
else:
    if x > 1:
        print('Positive')
    else:    
        print('Negative')

###
# for
###
# https://www.w3schools.com/python/python_for_loops.asp

# prints out the numbers 0,1,2,3,4
for x in range(0,5):
    print(x)
    
# prints out 3,4,5
for x in range(3, 6):
    print(x)

# prints out 3,5,7
for x in range(3, 8, 2):
    print(x)
    
# for loop ... strange!!!
for i in range(10):
    print(i)
    i = 5  
    print(i)

# 
for x in range(10, 0, -1):
    print(x)


###
# while
###
# https://www.w3schools.com/python/python_while_loops.asp

# while loop
count = 0
while count < 5:
    print(count)
    count += 1  # This is the same as count = count + 1

# while loop
count = 0
while count < 10:
    print(count)
    count += 1  # This is the same as count = count + 1
    count = 6

# prints out 0,1,2,3,4
count = 0
while True:
    print(count)
    count += 1
    if count >= 5:
        break
    print("More To Come ...")
    
# prints out only odd numbers - 1,3,5,7,9
for x in range(10):
    # Check if x is even
    if x % 2 == 0:
        continue
    print(x)
    
# 
count = 0
while count < 10:
    # Check if x is even
    if count % 2 == 0:
        count += 1  
        continue
    print(count)
    count += 1  # This is the same as count = count + 1
    
# 
count = 0
while count < 10:
    # Check if x is even
    if count % 2 == 1:
        print(count)
    count += 1  # This is the same as count = count + 1
    
