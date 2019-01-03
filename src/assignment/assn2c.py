
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import re
import string
from scipy.cluster.hierarchy import linkage,dendrogram
from matplotlib import pyplot as plt
import scipy.spatial.distance as ssd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import codecs
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[2]:


from assn1_1 import tokenize_stem as ts
from assn1_1 import get_text



# In[3]:


s = "msc-plagiarism-assigment/ass1-"
l = ["1349","422","734","808","936","1019","1037","1046","1138","1147","202","211","321","440","505","532","541","606","743","817","826","909"]
filenames = [s+item+".txt" for item in l]    


# In[4]:


filenames[0]


# In[5]:


texts = []
for names in filenames:
    texts.append(get_text(names))


# In[6]:


vectorizer = TfidfVectorizer(tokenizer=ts,max_df=0.5,lowercase=True,smooth_idf=True)


# In[7]:


tfidf_model = vectorizer.fit_transform(texts)


# In[8]:


tfidf_model.shape


# In[9]:


svd_model = TruncatedSVD(n_components=4, algorithm='randomized', n_iter=200)


# In[10]:


svd_model.fit(tfidf_model)


# In[11]:


len(svd_model.components_)


# In[12]:


terms = vectorizer.get_feature_names()
type(terms)
#terms.index("�")


# In[13]:


for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
    print("\n")


# In[14]:


reduced_tfidf = svd_model.transform(tfidf_model)


# In[15]:


reduced_tfidf.shape


# In[16]:


from sklearn.metrics.pairwise import cosine_similarity


# In[17]:


cs = cosine_similarity(reduced_tfidf,reduced_tfidf)
ds = cosine_similarity(reduced_tfidf,reduced_tfidf)


# In[18]:


cs.shape


# In[41]:


for i in range(ds.shape[0]):
    for j in range(ds.shape[1]):
        cs[i][j] = round(cs[i][j],2)
for i in range(ds.shape[0]):
    for j in range(ds.shape[1]):
        ds[i][j] = 1-round(cs[i][j],2)


# In[48]:


ds.shape


# In[47]:


distArray = ssd.squareform(ds)
print(distArray.shape)
Z = linkage(distArray)
fig = plt.figure(figsize=(25, 10))
    
dn = dendrogram(Z,labels=l)
plt.show()
    
#dn = dendrogram(Z,labels=l)
plt.show()






