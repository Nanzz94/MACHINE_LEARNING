#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


movies.shape


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


movies.shape


# In[7]:


movies.head()


# In[8]:


movies.info()


# In[9]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.head()


# In[11]:


movies.info()


# In[12]:


movies.isnull().sum()


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.duplicated().sum()


# In[15]:


movies.iloc[0].genres


# In[16]:


import ast
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[ ]:





# In[17]:


movies['genres']=movies['genres'].apply(convert)


# In[ ]:





# In[18]:


movies['keywords']=movies['keywords'].apply(convert)


# In[19]:


movies.head()


# In[51]:


movies.cast[0]


# In[20]:


import ast
def convert3(obj):
    l=[]
    c=0
    for i in ast.literal_eval(obj):
        l.append(i['name'])
        c+=1
        if(c==3):
            break
    return l


# In[ ]:





# In[21]:


movies['cast']=movies['cast'].apply(convert3)


# In[22]:


movies['crew'][0]


# In[23]:


import ast
def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l


# In[24]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[25]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[26]:


movies.head()


# In[27]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[28]:


movies.head()


# In[29]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[30]:


movies.head()


# In[31]:


new_df=movies[['movie_id','title','tags']]


# In[32]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[33]:


new_df.head()


# In[34]:


pip install nltk


# In[35]:


import nltk


# In[36]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[37]:


def stem(text):
     y=[]
     for i in text.split():
         y.append(ps.stem(i))
     return " ".join(y)


# In[38]:


new_df['tags'].apply(stem)


# In[39]:


new_df['tags'][0]


# In[40]:


new_df['tags'].apply(lambda x:x.lower())


# In[41]:


#text vectorization
#methods:
#bags of words(used below)(remove stop words)
#tfidf
#wordvsvec
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')#(it returns sparse matrix)


# In[42]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[43]:


vectors[0]


# In[44]:


cv.get_feature_names_out()


# In[45]:


ps.stem('loveing')


# In[46]:


# noew we willl calculate distance between two movies using cosine methods and not by euclids method as it fails for large dimensional data
from sklearn.metrics.pairwise import cosine_similarity


# In[47]:


similarity=cosine_similarity(vectors)


# In[48]:


similarity[0]


# In[53]:


enumerate(distances)


# In[49]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[50]:


recommend('Batman Begins')

