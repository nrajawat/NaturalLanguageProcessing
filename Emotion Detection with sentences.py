#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import seaborn as sns


# In[4]:


import neattext.functions as nxf


# In[8]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[9]:


df = pd.read_csv("emotion_data.csv")


# In[10]:


df.head()


# In[11]:


df['Emotion'].value_counts()


# In[13]:


sns.countplot(x='Emotion',data=df)


# In[15]:


dir(nxf)


# In[17]:


df['Clean_Text']= df['Text'].apply(nxf.remove_userhandles)


# In[29]:


df['Clean_Text']= df['Clean_Text'].apply(nxf.remove_stopwords)


# In[30]:


#df['Clean_Text']= df['Clean_Text'].apply(nxf.remove_special_cha)


# In[31]:


Xfeatures = df['Clean_Text']
ylabels = df['Emotion']


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(Xfeatures, ylabels,test_size=0.3, random_state=42)


# In[33]:


from sklearn.pipeline import Pipeline


# In[34]:


pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])


# In[35]:


pipe_lr.fit(x_train, y_train)


# In[37]:


pipe_lr


# In[36]:


pipe_lr.score(x_test, y_test)


# In[38]:


ex1 = "This book is so interesting it made me happy"


# In[39]:


pipe_lr.predict([ex1])


# In[40]:


pipe_lr.predict_proba([ex1])


# In[42]:


pipe_lr.classes_


# In[44]:


import joblib
pipeline_file = open("emotion_classifier_pipe_lr_17_March_2023.pkl","wb")
joblib.dump(pipe_lr,pipeline_file)
pipeline_file.close()


# In[ ]:




