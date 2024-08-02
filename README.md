# Spam-mail-Detection: Machine Learning Model for Predicting Spam Mails with nearly 99% accuracy


## Introduction

Spam emails, which clutter inboxes with unsolicited and often malicious content, pose a significant challenge in digital communication. This project leverages machine learning techniques to predict whether an email is spam or not. By analyzing attributes such as the content of the email, the model aims to classify emails accurately into "spam" or "ham" (non-spam).

## Table of Contents
1. [Introduction](#introduction)
2. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
3. [Data Preparation](#Data-Preparation)
4. [Model Evaluation](#Evaluate-models)



#Exploratory Data Analysis

### Importing Necessary Libraries
Importing Necessary Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
```

##Load Spam Mail Dataset
Load the dataset Load the dataset into DataFrame.

```python
df = pd.read_csv('/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv')
```

## Display the first few rows of the dataset
inspect the first few rows to understand its structure.
```python
df.head()
```

![image](https://github.com/user-attachments/assets/d71c88b4-2e3a-4627-8c66-80c6c6c42301)


## How Many Instances and Features ?
Display the number of rows and columns in the dataset
```python
print("Shape of the dataframe:", df.shape)
```



##Display Variables DataType and count of non-NULL values
```python
df.info()
```
![image](https://github.com/user-attachments/assets/faebc405-efe6-4643-b569-3c2f798ff1fa)

All variables Have 0 NULLs



##Count of Spam and Ham mails
Show the Distribution of values in 'label' Column
```python
print(df['label'].value_counts())
```
it Shows that it is balanced 






# Data Preparation
## Data Transformation
Convert categorical variable 'label' into numerical one for machine learning model.

```python
#we will convert it into 1 for ham and 0 for spam
df.replace({'label': {'ham': 1, 'spam': 0}}, inplace=True)
```


#Now Let's Drop Columns that are not important like 'label_num' since it will introduce no information and 'Unnamed: 32' since it have only NULLs
```python
df.drop(['label_num','Unnamed: 32'] , axis=1 , inplace=True)
```


## Split data into Features and Target sets

```python
x = df['text']
y = df['label']
```


## Split data into training, Validation and testing sets

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Print the shapes of each set to verify the splits
print("Training set:", x_train.shape, y_train.shape)
print("Testing set:", x_test.shape, y_test.shape)


```


##Feature Extraction
Convert text into numerical values using TfidfVectorizer

```python
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = tfidf_vectorizer.fit_transform(x_train)
x_test_features = tfidf_vectorizer.transform(x_test)
```


## Evaluate models
after training the model and predicting it on test data it makes accuracy of nearly 99% Using LogisticRegression Machine Learning Algorithm

##Creating Confusion Matrix
it shows the predicted values Distribution

```python
cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();
```
![image](https://github.com/user-attachments/assets/84a8e1db-33bc-4586-9351-a3523efe427d)
