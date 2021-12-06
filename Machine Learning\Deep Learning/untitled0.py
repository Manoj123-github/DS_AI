# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:23:09 2021

@author: Manoj Yadav
"""

import re
import traceback
from deepface import DeepFace
import random 
random.seed(3)
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
import matplotlib.image as mpimg
import os
import glob
import pandas as pd
import numpy as np


os.getcwd()


## Total images in the folder
img_len=len(glob.glob('C:/Users/Manoj Yadav/Desktop/Project/human data/*'))

print(img_len)


# Define a function for Age and Gender Prediction
# Deepface uses separate pretrained weight for Age and Gender.
#  Define the objective in the action argument of DeepFace.analyze() methos
## Get the gender from the loop
## The functio iterates over the images and uses the deepface age and gender pre trained weights to predict the age and gender.
## Dict is storing Gender,Age and agebucket with key as image name
def calculate_gender(image,img_name):
    name={}
    gender=[]
    age=[]
    agebucket=[]
    try:
        img_arr=cv2.imread(image)
        ## get gender
        response=DeepFace.analyze(img_arr,actions=["gender","age"],enforce_detection=False)
        gender.append(response['gender'])
        age.append(response['age'])
        ## Bucket the age
        if int(age[0])>=13 and int(age[0])<=17:
            agebucket.append('13-17years')
        elif int(age[0])>17 and int(age[0])<=24:
            agebucket.append('18-24years')
        elif int(age[0])>24 and int(age[0])<=34:
            agebucket.append('25-34years')
        elif int(age[0])>34 and int(age[0])<=44:
            agebucket.append('35-44years')
        elif int(age[0])>44 and int(age[0])<=54:
            agebucket.append('45-54years')
        elif int(age[0])>54 and int(age[0])<=64:
            agebucket.append('55-64years')
        elif int(age[0])>64:
            agebucket.append('above 65years')
        else:
            agebucket.append('NA')
        ## store in dictionary
        name[img_name]=(gender[0],age[0],agebucket[0])
    except:
        name[img_name]='NA' ## If the image is not a front facing image
        traceback.print_exc()
    return name

#Iterate through the sample images to predict the gender and age
## Iterate through the image to calculate the gender: Pick only 50 
count=0
img_list=[]
for i in range(img_len):
    image=glob.glob("C:/Users/Manoj Yadav/Desktop/Project/human data/*")[i]
    print(str(image[7:]))
    count+=1
    img_list.append(calculate_gender(image,str(glob.glob("C:/Users/Manoj Yadav/Desktop/Project/human data/*")[i][7:])))
    if count<50:
        continue
    else:
        break
    
img_list[:10]


# create figure
fig = plt.figure(figsize=(15, 15))

# setting values to rows and column variables
rows = 6
columns = 3

for i in range(18):
    image=glob.glob("/kaggle/input/human-faces/Humans/*")[i]
    img_arr=cv2.imread(image)
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, i+1)
    # showing image
    plt.imshow(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image:"+glob.glob("C:/Users/Manoj Yadav/Desktop/Project/human data/*")[i][7:]+"\n"+"Gender:"
                      +img_list[i][glob.glob("C:/Users/Manoj Yadav/Desktop/Project/human data/*")[i][7:]][0]+" "
                      +"Age:"+str(img_list[i][glob.glob("C:/Users/Manoj Yadav/Desktop/Project/human data/*")[i][7:]][1])+" "
                      +"Age Bucket:"+img_list[i][glob.glob("C:/Users/Manoj Yadav/Desktop/Project/human data/*")[i][7:]][2])
    

