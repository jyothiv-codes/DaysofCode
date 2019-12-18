#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("train_NIR5Yl1.csv")
df.head(3)


# In[2]:


df.isnull().sum()


# In[3]:


df1 = df[df.isna().any(axis=1)]


# In[4]:


print(df1)


# In[5]:


df1 = df.groupby('Tag') 


# In[6]:


df3=df1.get_group('x') 


# In[7]:


df2 = df[df.isna().any(axis=1)]


# In[8]:


print(df2)


# In[9]:


df4 = df.groupby('ID') 


# In[10]:


df2=df4.get_group(268110) 
print(df2)


# In[11]:


from statistics import mean
views=df3['Views'].mean()
print(views)


# In[12]:


upvotes=df3['Upvotes'].mean()
print(upvotes)


# In[13]:


df['Views']=df['Views'].fillna(views)
df['Upvotes']=df['Upvotes'].fillna(upvotes)


# In[14]:


df['Username'].unique()


# In[15]:


min(df['Username'])


# In[16]:


df['Username']=df['Username'].fillna(1)


# In[17]:


df.isnull().sum()


# In[18]:


import pandas as pd

one_hot = pd.get_dummies(df['Tag'])
df = df.drop('Tag',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[19]:


dfcorr.corr(method ='pearson') 


# In[20]:


import pandas as pd
import numpy as np


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r' & 'BrBG' are other good diverging colormaps



# In[21]:


from sklearn import linear_model

X = df.drop(['a','c','Upvotes','h','i','j','o','p','r','s','x'],axis=1)
y = df['Upvotes']


# In[158]:


import pandas as pd
from sklearn import preprocessing

x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)


# In[22]:


# Normalize the data attributes for the Iris dataset.
#from sklearn.datasets import load_iris
from sklearn import preprocessing
# load the iris dataset


# normalize the data attributes
normalized_X = preprocessing.normalize(X)


# In[35]:


lm = linear_model.LinearRegression()
model = lm.fit(normalized_X,y)


# In[36]:


test=pd.read_csv("test_8i3B3FC.csv")


# In[37]:


test.head(2)


# In[38]:


test.isnull().sum()


# In[39]:


import pandas as pd

one_hot = pd.get_dummies(test['Tag'])
test =test.drop('Tag',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[40]:



testx = test.drop(['a','c','h','i','j','o','p','r','s','x'],axis=1)


# In[42]:


# Normalize the data attributes for the Iris dataset.

from sklearn import preprocessing
# load the iris dataset


# normalize the data attributes
normalized_X = preprocessing.normalize(testx)


# In[43]:


predictions = lm.predict(normalized_X)
print(predictions)


# In[44]:


submission=pd.read_csv("sample_submission_OR5kZa5.csv")


# In[45]:


submission.shape


# In[46]:


submission.columns


# In[47]:


ids=submission['ID']


# In[48]:


submission=submission.iloc[0:0]
submission['ID']=ids


# In[49]:


submission['Upvotes']=predictions 


# In[50]:


submission.head(4)


# In[ ]:





# In[51]:


pd.DataFrame(submission, columns=['ID','Upvotes']).to_csv('upvotes.csv')


# In[ ]:





# In[ ]:




