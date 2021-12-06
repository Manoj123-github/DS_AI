
@url: https://matplotlib.org/users/pyplot_tutorial.html
@url: https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python#gs.NClMOE0
@url: https://pythonprogramming.net/matplotlib-python-3-basics-tutorial/
"""

# Anatomy Of A Plot
# The Canvas / Figure is the overall window or page that everything is drawn on. 
# It's the top-level component of all the ones that you will consider in the 
# following points. You can create multiple independent Figures. 
# A Figure can have several other things in it, such as a subtitle, which is 
# a centered title to the figure. You'll also find that you can add a legend 
# and color bar to your Figure.
# Axes - most plotting ocurs on an Axes. The axes is the area on which the 
# data is plotted and that can have ticks, labels, etc. associated with it. 
# This explains why Figures can contain multiple Axes.
# Each Plot has an x-axis and a y-axis, which contain ticks, which have major 
# and minor ticklines and ticklabels. There’s also the axis labels, title, and 
# legend to consider when you want to customize your axes, but also taking into 
# account the axis scales and gridlines might come in handy.
# Spines are lines that connect the axis tick marks and that designate the 
# boundaries of the data area. In other words, they are the simple black 
# square that you get to see when you don’t plot any data at all but when you 
# have initialized the Axes.

# matplotlib
# gallery: http://matplotlib.org/index.html

# import the necessary packages and modules
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

# prepare the data
x = np.linspace(0, 10, 100)
print(x)
# init plot
plt.figure()
# plot the data
plt.plot(x, x)
# show the plot
plt.show()


# simple plot
# In this section, we want to draw the cosine and sine functions on the same 
# plot. Starting from the default settings, we'll enrich the figure step by 
# step to make it nicer.
# First step is to get the data for the sine and cosine functions:
# import the necessary packages and modules
# prepare data
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
print(X)
C = np.cos(X)
S = np.sin(X)
# prepare plot
plt.figure()
plt.plot(X,C)
plt.plot(X,S)
# show plot
plt.show()

# note
# understand how two plots are being drawn on the same canvas / figure
# both are required if legend is to be shown
# understand the significance of .show

# now backto basics
# line chart
# prepare plot
plt.figure()
plt.plot([1,2,3],[4,5,1])
# show plot
plt.show()

# note
# use .plot to draw line graph

# more of basics
# line chart
# import the necessary packages and modules
# prepare data
x = [5,8,10]
y = [12,16,6]
# prepare plot
plt.figure()
plt.plot(x, y, label='data')
# annotations
plt.title('Graph Title')
plt.xlabel('X axis')
plt.ylabel('Y axis')
# add a legend
plt.legend()
# show plot
plt.show()

# note
# importance of label & legend
# both are required if legend is to be shown

# style
# line chart
# import the necessary packages and modules
import matplotlib.style as style
# use style
style.use('ggplot')
# prepare data
x1 = [5,8,10]
y1 = [12,16,6]
x2 = [6,9,11]
y2 = [6,15,7]
# prepare plot
plt.figure()
plt.plot(x1,y1,label='series 1')
plt.plot(x2,y2,label='series 2')
# annotations
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
# show plot
plt.grid(True,color='k')
# add a legend
plt.legend()
plt.show()

# note
# understand use of style.use
# understand use of .grid

# color, marker, linestyle, linewidth
# line chart
# prepare data
x1 = [5,8,10]
y1 = [12,16,6]
x2 = [6,9,11]
#x2 = [5,8,10]
y2 = [6,15,7]
#y2 = [12,16,6]
# prepare plot
plt.figure()
plt.plot(x1,y1,color='indianred', label='line one', marker='*', linestyle='dotted', linewidth=1)
plt.plot(x2,y2,color='c', label='line two', marker='*', linestyle='-', linewidth=1)
# annotations
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
# show plot
plt.legend()
plt.show()

# note
# understand use of marker, linestyle, linewidth

# barchart
# prepare data
x1 = [5,8,10]
y1 = [12,16,6]
# prepare plot
plt.figure()
plt.bar(x1,y1,label='series1')
# annotations
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.legend()
# show plot
plt.show()

# barchart
# prepare data
x1 = [5,8,10]
y1 = [12,16,6]
x2 = [5,8,10]
#x2 = [6,9,11]
y2 = [6,15,7]
# prepare plot
plt.figure()
plt.bar(x1,y1,label='series1')
plt.bar(x2,y2,label='series2')
#plt.bar(x1,y1,color='b',label='series1')
#plt.bar(x2,y2,color='b',label='series2')
# annotations
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()
# show plot
plt.show()

# note
# use .plot to draw line graph
# important
# what happens in above when y1 & y2 are same
# or 
# some elements of y1 & y2 are same
# also important
# why different series / plots are displyed in different colors
# color handling to use non defualt colors
# color handling to use same colors

# scatter
# prepare data
x1 = [5,8,10]
y1 = [12,16,6]
x2 = [5,8,10]
y2 = [12,16,6]
#x2 = [6,9,11]
#2 = [6,15,7]
# prepare plot
plt.figure()
#plt.scatter(x1,y1)
#plt.scatter(x2,y2)
plt.scatter(x1,y1,color='r')
plt.scatter(x2,y2,color='b')
# annotations
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
# show plot
plt.show()

# note
# use .scatter to draw scatter plot
# important
# what happens in above when x1,y1 are same as x2,y2
# also important
# why different series / plots are displyed in different colors
# color handling to use non defualt colors
# color handling to use same colors

# histogram
# prepare data
ages = [22,55,62,15,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]
bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]
# prepare plot
plt.figure()
#plt.hist(ages, bins, histtype='step', color='b', rwidth=2)
plt.hist(ages, bins, histtype='bar', color='b', rwidth=2)
# annotations
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()
plt.show()

# note
# use .hist to draw histogram
# reply histtype="step" / "bar"


# barchart with string in x-axis
# prepare data
#x1 = ['01', '02', '03', '04', '05', '11', '12', '13', '14', '15']
x1 = ['bb', 'cc', 'aa', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj']
y1 = [173, 135, 141, 148, 140, 149, 152, 178, 135, 96]
# prepare plot
plt.figure()
plt.bar(x1, y1)
#plt.bar(range(len(x1)), y1)
#plt.xticks(range(len(x1)), x1)
# annotations
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
# show plot
plt.show()

# note
# bar chart requires numerci data in x-axis & y-axis
# how to display graphs is x-axis is alpha-numeric

# barchart with date in x-axis
# import the necessary packages and modules
import datetime
x1 = [datetime.datetime(2017, 1, 1),
    datetime.datetime(2017, 1, 2),
    datetime.datetime(2017, 1, 3)]
#x1 = [datetime.datetime(2017, 1, 1),
#    datetime.datetime(2017, 2, 1),
#    datetime.datetime(2017, 3, 1)]
y1 = [4, 9, 2]
# prepare plot
plt.figure()
plt.bar(x1, y1)
# annotations
plt.title('Graph Title')
plt.xlabel('X axis')
# show plot
plt.show()

# note
# bar chart requires numerci data in x-axis & y-axis
# how to display graphs is x-axis is date-time

# line chart with string in x-axis
# prepare data
#x1 = ['01', '02', '03', '04', '05', '11', '12', '13', '14', '15']
x1 = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj']
y1 = [173, 135, 141, 148, 140, 149, 152, 178, 135, 96]
# prepare plot
plt.figure()
plt.plot(x1, y1)
#plt.xticks(range(len(x1)), x1)
# annotations
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
# show plot
plt.show()

# note
# line chart requires numerci data in x-axis & y-axis
# how to display graphs is x-axis is alpha-numeric

# line chart with date in x-axis
# prepare data
x1 = [datetime.datetime(2017, 1, 1),
    datetime.datetime(2017, 1, 2),
    datetime.datetime(2017, 1, 3)]
#x1 = [datetime.datetime(2017, 1, 1),
#    datetime.datetime(2017, 2, 1),
#    datetime.datetime(2017, 3, 1)]
y1 = [4, 9, 2]
# prepare plot
plt.figure()
plt.plot(x1, y1)
# annotations
plt.title('Graph Title')
plt.xlabel('X axis')
# show plot
plt.show()

# note
# line chart requires numerci data in x-axis & y-axis
# how to display graphs is x-axis is date-time

# read data
# import 
import pandas as pd
# prepare data
# create dataframe
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
        'female': [0, 1, 1, 0, 1],
        'age': [42, 52, 36, 24, 73],
        'preTestScore': [4, 24, 31, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'female', 'preTestScore', 'postTestScore'])
print(df)

# scatter plot using dataframe
# prepare plot
plt.figure()
plt.scatter(df.age, df.preTestScore,label='PreTest')
plt.scatter(df.age, df.postTestScore,label='PostTest')
# annotation
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()
# show plot
plt.show()

# note
# funda remains the same
# use .scatter with x & y

# line plot using dataframe
# prepare plot
plt.figure()
plt.plot(df.age, df.preTestScore,label='PreTest')
plt.plot(df.age, df.postTestScore,label='PostTest')
# annotation
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()
# show plot
plt.show()

# note
# funda remains the same
# use .plot with x & y

# bar plot using dataframe
# prepare plot
plt.figure()
plt.bar(df.first_name, df.age)
#plt.bar(df.index, df.age)
#plt.xticks(df.index, df.first_name)
# annotation
plt.title('Graph Title')
plt.ylabel('Y axis')
plt.xlabel('X axis')
# show plot
plt.show()

# note
# funda remains the same
# use .bar with x & y
# also
# see use of df.index & plt.xticks