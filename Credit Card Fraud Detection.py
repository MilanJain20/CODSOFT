#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv("H:\\moon\\OneDrive\\Desktop\\creditcard.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[7]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


# distribution of legal transaction and fraud transaction
df['Class'].value_counts()


# # This data is very unbalanced.
# 0 -> legal or normal transaction
# 1 -> fraud transaction

# In[14]:


#separating data for analysis
legal=df[df.Class==0]
fraud=df[df.Class==1]


# In[17]:


fraud.shape


# In[16]:


legal.shape


# In[19]:


# statistical measures for the data 
legal.Amount.describe()


# In[20]:


fraud.Amount.describe()


# In[21]:


df.groupby('Class').mean()


# In[ ]:


# Build a sample dataset conatining similar distribution of legal transaction and fraud transaction


# In[22]:


legal_sample= legal.sample(n=492)


# In[ ]:


# Concatenating two data frames


# In[24]:


new_df=pd.concat([legal_sample , fraud] , axis=0)


# In[25]:


new_df.head()


# In[26]:


new_df.tail()


# In[28]:


new_df.info()


# In[29]:


new_df.isnull().sum()


# In[30]:


new_df['Class'].value_counts()


# In[31]:


new_df.groupby('Class').mean()


# In[ ]:


# splitting the data into tagets and features


# In[34]:


X=new_df.drop(columns='Class' , axis=1)
Y=new_df['Class']
print(X)


# In[35]:


print(Y)


# In[37]:


# split the into training data and testing data
X_train , X_test , Y_train , Y_test= train_test_split(X ,Y,test_size=0.2 , stratify=Y , random_state=2)


# In[39]:


print(X.shape , X_train.shape , X_test.shape)


# # Model Training by Logical regression

# In[40]:


model =LogisticRegression()


# In[41]:


model.fit(X_train , Y_train)


# # Model Evaluation --> Accuracy Score

# In[42]:


# Accuracy on training data
X_Train_prediction = model.predict(X_train)
training_data_accuracy= accuracy_score(X_Train_prediction , Y_train)
print("Accuracy on training data : " , training_data_accuracy)


# In[44]:


# Accuracy on testing data
X_test_pediction=model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_pediction , Y_test)
print("Accuracy on testing data : " , testing_data_accuracy)

