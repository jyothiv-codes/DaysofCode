#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd
df=pd.read_csv("train_NIR5Yl1.csv")
df.head(3)


# In[91]:


df.isnull().sum()


# In[92]:


df1 = df[df.isna().any(axis=1)]


# In[93]:


print(df1)


# In[94]:


df1 = df.groupby('Tag') 


# In[95]:


df3=df1.get_group('x') 


# In[96]:


df2 = df[df.isna().any(axis=1)]


# In[97]:


print(df2)


# In[98]:


df4 = df.groupby('ID') 


# In[99]:


df2=df4.get_group(268110) 
print(df2)


# In[100]:


from statistics import mean
views=df3['Views'].mean()
print(views)


# In[101]:


upvotes=df3['Upvotes'].mean()
print(upvotes)


# In[102]:


df['Views']=df['Views'].fillna(views)
df['Upvotes']=df['Upvotes'].fillna(upvotes)


# In[103]:


df['Username'].unique()


# In[104]:


min(df['Username'])


# In[105]:


df['Username']=df['Username'].fillna(1)


# In[106]:


df.isnull().sum()


# In[107]:


import pandas as pd

one_hot = pd.get_dummies(df['Tag'])
df = df.drop('Tag',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[110]:


dfcorr.corr(method ='pearson') 


# In[111]:


import pandas as pd
import numpy as np


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r' & 'BrBG' are other good diverging colormaps



# In[112]:


from sklearn import linear_model

X = df.drop(['a','c','Upvotes','h','i','j','o','p','r','s','x'],axis=1)
y = df['Upvotes']


# In[113]:


lm = linear_model.LinearRegression()
model = lm.fit(X,y)


# In[114]:


test=pd.read_csv("test_8i3B3FC.csv")


# In[115]:


test.head(2)


# In[116]:


test.isnull().sum()


# In[117]:


import pandas as pd

one_hot = pd.get_dummies(test['Tag'])
test =test.drop('Tag',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[118]:



testx = test.drop(['a','c','h','i','j','o','p','r','s','x'],axis=1)


# In[119]:


predictions = lm.predict(testx)
print(predictions)


# In[120]:


submission=pd.read_csv("sample_submission_OR5kZa5.csv")


# In[121]:


submission.shape


# In[122]:


submission.columns


# In[123]:


ids=submission['ID']


# In[124]:


submission=submission.iloc[0:0]
submission['ID']=ids


# In[125]:


submission['Upvotes']=predictions 


# In[126]:


submission.head(4)


# In[ ]:





# In[127]:


pd.DataFrame(submission, columns=['ID','Upvotes']).to_csv('upvotes.csv')


# In[ ]:




