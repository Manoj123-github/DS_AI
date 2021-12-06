
# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 5)
# sns
import seaborn as sns
# util
#import utils

##############################################################
# Read Data 
##############################################################

# read file
print("\n*** Read Data ***")
df = pd.read_csv("./data/smscollection.dat", sep='\t', encoding = 'latin1', header=None, names=['Label', 'SMS'])
print("Done ...")


##############################################################
# Exploratory Data Analytics
##############################################################

# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())


##############################################################
# Class Variable & Counts
##############################################################

# store class variable  
# change as required
clsVars = "Label"
print("\n*** Class Vars ***")
print(clsVars)

# change as required
txtVars = "SMS"
print("\n*** Text Vars ***")
print(txtVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# label counts ... anpther method
print("\n*** Label Counts ***")
print(df['Label'].value_counts())


##############################################################
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
#df = df.drop('Id', axis=1)
print("None ...")


##############################################################
# Visual Data Anlytics
##############################################################

# check class
# outcome groupby count    
print("\n*** Group Counts ***")
print(df.groupby(clsVars).size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df[clsVars],label="Count")
plt.title('Class Variable')
plt.show()

# plotting graph ham frequent words
from collections import Counter
print("\n*** Frequent Words - Ham ***")
vHamCount = Counter(" ".join(df[df['Label']=='ham']['SMS']).split()).most_common()
dfHamCount = pd.DataFrame.from_dict(vHamCount)
dfHamCount = dfHamCount.rename(columns={0: "Word", 1 : "Freq"})
dfHamCount = dfHamCount[dfHamCount['Word'].apply(lambda x: len(str(x))>3)]
print(dfHamCount.head(10))
print("Done ...")

print("\n*** Frequent Words - Spam ***")
vSpamCount = Counter(" ".join(df[df['Label']=='spam']['SMS']).split()).most_common()
dfSpamCount = pd.DataFrame.from_dict(vSpamCount)
dfSpamCount = dfSpamCount.rename(columns={0: "Word", 1 : "Freq"})
dfSpamCount = dfSpamCount[dfSpamCount['Word'].apply(lambda x: len(str(x))>3)]
print(dfSpamCount.head(10))
print("Done ...")

# plot horizontal bar - top 10 ham
print("\n*** Ham Top 10 Frequent Words ***")
dft = dfHamCount[0:9]
plt.figure()
sns.barplot(x="Freq", y="Word", data=dft, color="b", orient='h')
plt.show()

# plot horizontal bar - top 10 spam
print("\n*** Spam Top 10 Frequent Words ***")
dft = dfSpamCount[0:9]
plt.figure()
sns.barplot(x="Freq", y="Word", data=dft, color="r", orient='h')
plt.show()

################################
# Classification 
# Split Train & Test
###############################

# columns
print("\n*** Columns ***")
X = df[txtVars].values
y = df[clsVars].values
print("Class: ",clsVars)
print("Text : ",txtVars)

# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                test_size=0.33, random_state=707)

# print
print("\n*** Length Of Train & Test Data ***")
print("X_train: ", len(X_train))
print("X_test : ", len(X_test))
print("y_train: ", len(y_train))
print("y_test : ", len(y_test))

# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))

################################
# Classification 
# Count Vectorizer
###############################

# convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
print("\n*** Count Vactorizer Model  ***")
cv = CountVectorizer(max_features = 1500)
print(cv)
cv.fit(X_train)
print("Done ...")

# count vectorizer for train
print("\n*** Count Vectorizer For Train Data ***")
X_train_cv = cv.transform(X_train)
print(X_train_cv[0:4])

print("\n*** Count Vectorizer For Test Data ***")
X_test_cv = cv.transform(X_test)
print(X_test_cv[0:4])


################################
# Classification 
# actual model ... create ... fit ... predict
###############################

# imports
print("\n*** Imports ***")
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print("Done ...")

print("\n*** Create Model ***")
# create model
model = MultinomialNB(alpha = 0.5)
model.fit(X_train_cv,y_train)
print("Done ...")

# predict
print("\n*** Predict Test Data ***")
p_test = model.predict(X_test_cv)
print("Done ...")

# accuracy
accuracy = accuracy_score(y_test, p_test)*100
print("\n*** Accuracy ***")
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
cm = confusion_matrix(y_test, y_test)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix
# X-axis Predicted | Y-axis Actual
cm = confusion_matrix(y_test, p_test)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_test,p_test)
print(cr)

################################
# Final Prediction
# Create model Object from whole data
# Read .prd file
# Predict class
# Confusion matrix with data in .prd file
###############################

# classifier object
print("\n*** Classfier Object ***")
cv = CountVectorizer(max_features = 1500)
print(cv)
cv.fit(X)
X_cv = cv.transform(X)
print("Done ...")

print("\n*** Create Model ***")
# create model
model = MultinomialNB(alpha = 0.5)
model.fit(X_cv,y)
print("Done ...")

# read dataset
print("\n*** Read Data For Prediction ***")
dfp = pd.read_csv("./data/smscollection-prd.dat", sep='\t', encoding = 'latin1', header=None, names=['Label', 'SMS'])
print(dfp.head())

# prepare data
print("\n*** Prepare Predit Data ***")
X_pred = dfp[txtVars].values
y_pred = dfp[clsVars].values
print("Done ...")

# predict
print("\n*** Predict Test Data ***")
X_pred_cv = cv.transform(X_pred)
p_pred = model.predict(X_pred_cv)
print("Done ...")

# accuracy
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_pred, p_pred)*100
print(accuracy)

# confusion matrix - actual
cm = confusion_matrix(y_pred, y_pred)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix - predicted
cm = confusion_matrix(y_pred, p_pred)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_pred, p_pred)
print(cr)
