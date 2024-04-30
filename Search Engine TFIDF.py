#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3


conn = sqlite3.connect('data/eng_subtitles_database.db')

cursor = conn.cursor()

query = "SELECT name FROM sqlite_master WHERE type='table';"

# Execute the query
cursor.execute(query)

# Fetch all the results
tables = cursor.fetchall()

# Close the cursor and the connection
cursor.close()
conn.close()

# Extract table names from the fetched results
table_names = [table[0] for table in tables]

# Display the table names
print("Table names:")
for name in table_names:
    print(name)


# In[2]:


import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('data/eng_subtitles_database.db')

# Query to select all data from a specific table (replace 'your_table_name' with the actual table name)
query = "SELECT * FROM zipfiles;"

# Use pandas to read the data into a DataFrame
df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Now you can work with the DataFrame 'df' as needed
df.head()  # Display the first few rows of the DataFrame


# In[3]:


df


# In[4]:


import zipfile
import io

count = 0

def decode_method(binary_data):
    global count
    # Decompress the binary data using the zipfile module
    # print(count, end=" ")
    count += 1
    with io.BytesIO(binary_data) as f:
        with zipfile.ZipFile(f, 'r') as zip_file:
            # Assuming there's only one file in the ZIP archive
            subtitle_content = zip_file.read(zip_file.namelist()[0])
    
    # Now 'subtitle_content' should contain the extracted subtitle content
    return subtitle_content.decode('latin-1')  # Assuming the content is UTF-8 encoded text


# In[5]:


df['file_content'] = df['content'].apply(decode_method)

df.head()


# In[6]:


from bs4 import BeautifulSoup
import re


# In[7]:


raw_text = df['file_content'][32]
rhtml = BeautifulSoup(raw_text, 'html.parser').get_text()


# In[8]:


rhtml.split('\n')


# In[9]:


sentence = re.sub("[^a-zA-Z.]", " ", rhtml).strip()


# In[10]:


sentence


# In[11]:





# In[13]:


tokens = sen.split('.')


# In[14]:


tokens


# In[15]:


df


# In[17]:


df.shape


# In[19]:


df.sample(frac=0.3).to_csv('data/data_frame_searchengine.csv', index=False, escapechar='\\')


# In[ ]:




