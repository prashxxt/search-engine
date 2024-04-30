#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv(p"//search_eng.csv')


# In[3]:


df.head()


# In[4]:


df.clean_text_lemma.isnull().sum()


# In[5]:


df.sample(50)


# In[6]:


# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(df.clean_text_lemma)


# In[7]:


# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Concatenate the original DataFrame with the TF-IDF DataFrame
df = pd.concat([df, tfidf_df], axis=1)


# In[8]:


df


# In[9]:


df.columns


# In[10]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:





# In[15]:


from sklearn.metrics.pairwise import cosine_similarity

def filter_similar_records(query, df, tfidf_vectorizer, tfidf_matrix, threshold=None, top_n=None):
    # Transform the query using the same TF-IDF vectorizer
    query_tfidf = tfidf_vectorizer.transform([query])

    # Calculate cosine similarity between the query and all documents
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Add cosine similarity scores as a new column in the DataFrame
    df['cosine_similarity'] = cosine_similarities

    # Sort the DataFrame based on cosine similarity scores
    df = df.sort_values(by='cosine_similarity', ascending=False).reset_index(drop=True)

    # Filter based on threshold similarity score or select top N similar records
    if threshold is not None:
        filtered_df = df[df['cosine_similarity'] >= threshold]
    elif top_n is not None:
        filtered_df = df.head(top_n)
    else:
        filtered_df = df

    return filtered_df


# In[27]:


# Example usage:
query = input('Search : ')
# Lowering the threshold initially to get some results
threshold = 0.1
top_n = int(input('Enter the number of top similar records to retrieve: '))

filtered_records = filter_similar_records(query, df, tfidf_vectorizer, tfidf_matrix, threshold=threshold, top_n=top_n)
print(f"Filtered records for query '{query}':")
filtered_records[['name','num', 'cosine_similarity']]


# In[19]:





# In[21]:


df


# In[23]:


df[['cosine_similarity']]


# In[ ]:




