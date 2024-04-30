#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('p//datasearchengine.csv')


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df_new = df[['num', 'name', 'file_content']]


# In[6]:


df_new


# In[7]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[8]:


lemmatizer = WordNetLemmatizer()


# In[9]:


def preprocess(raw_text):
    # Remove HTML tags
    text = BeautifulSoup(raw_text, 'html.parser').get_text()
    
    # Removing special characters and digits
    special_char = re.sub("[^a-zA-Z]", " ", text)
    
    # change sentence to lower case
    lowered = special_char.lower()
    
    
    sentence = re.sub('\s+', ' ', lowered)

    tokens = sentence.split()
    
    # Lemmatization
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return pd.Series([" ".join(clean_tokens), len(clean_tokens)])


# In[10]:


from tqdm import tqdm, tqdm_notebook
import re
from bs4 import BeautifulSoup


# In[11]:





# In[12]:


temp_df = df_new['file_content'].progress_apply(lambda x: preprocess(x))

temp_df.head()


# In[13]:


temp_df.columns = ['clean_text_lemma', 'text_length_lemma']

temp_df.head()


# In[14]:


df_new = pd.concat([df_new, temp_df], axis=1)
df_new.head()


# In[18]:


df_10 = df_new.sample(frac=0.1)


# In[19]:


df_10


# In[22]:


df_10.sample(50)


# In[23]:


df_20 = df_new.sample(frac=0.2)


# In[24]:


# df_20.to_csv('data/search_eng_data_20_percent.csv', index=False, escapechar="\\")


# In[25]:


df_20


# In[ ]:




