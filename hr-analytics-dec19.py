#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
df=pd.read_csv("train_LZdllcl.csv")
df.head(3)


# In[90]:


df.isnull().sum()


# In[91]:


df['education'].unique()


# In[92]:


df['previous_year_rating'].unique()


# In[93]:


meanrating=df['previous_year_rating'].mean()


# In[94]:


print(meanrating)


# In[95]:


df['previous_year_rating']=df['previous_year_rating'].fillna(meanrating)


# In[96]:


d={}
for i in df['education']:
    d[i]=d.get(i,0)+1
print(d)


# In[97]:


df['education']=df['education'].fillna("Master's & above")


# In[98]:


df.isnull().sum()


# In[99]:


df=df.drop('employee_id',axis=1)


# In[100]:


import pandas as pd

one_hot = pd.get_dummies(df['education'])
df = df.drop('education',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[101]:


import pandas as pd

one_hot = pd.get_dummies(df['department'])
df = df.drop('department',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[102]:


import pandas as pd

one_hot = pd.get_dummies(df['region'])
df = df.drop('region',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[103]:


import pandas as pd

one_hot = pd.get_dummies(df['gender'])
df = df.drop('gender',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[104]:


import pandas as pd

one_hot = pd.get_dummies(df['recruitment_channel'])
df = df.drop('recruitment_channel',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[ ]:


X=df.drop('is_promoted',axis=1)
y=df['is_promoted']


# In[135]:


from sklearn import preprocessing
X = preprocessing.normalize(X)


# In[105]:


test=pd.read_csv("test_2umaH9m.csv")


# In[106]:


test.isnull().sum()


# In[107]:


meanrating=test['previous_year_rating'].mean()


# In[108]:


test['previous_year_rating']=test['previous_year_rating'].fillna(meanrating)


# In[109]:


d={}
for i in test['education']:
    d[i]=d.get(i,0)+1
print(d)


# In[110]:


test['education']=test['education'].fillna("Bachelor's")


# In[111]:


test.isnull().sum()


# In[112]:


import pandas as pd

one_hot = pd.get_dummies(test['department'])
test = test.drop('department',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[113]:


import pandas as pd

one_hot = pd.get_dummies(test['region'])
test = test.drop('region',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[114]:


import pandas as pd

one_hot = pd.get_dummies(test['gender'])
test= test.drop('gender',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[115]:


import pandas as pd

one_hot = pd.get_dummies(test['recruitment_channel'])
test = test.drop('recruitment_channel',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[116]:


import pandas as pd

one_hot = pd.get_dummies(test['education'])
test = test.drop('education',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[117]:


testx=test.drop('employee_id',axis=1)


# In[134]:


from sklearn import preprocessing
testx = preprocessing.normalize(testx)


# In[136]:


import sklearn as sk
from sklearn import svm
import pandas as pd
import os


SVM = svm.SVC(kernel='linear')
SVM.fit(X, y)
predictions=SVM.predict(testx)
round(SVM.score(X,y), 4)


# In[137]:


print(predictions)


# In[ ]:





# In[138]:


submission=pd.read_csv("sample_submission_M0L0uXE.csv")


# In[139]:


submission.columns


# In[141]:


ids=submission['employee_id']
submission=submission.iloc[0:0]
submission['employee_id']=ids
submission.head(4)
submission['is_promoted']=predictions 


pd.DataFrame(submission, columns=['employee_id','is_promoted']).to_csv('hr.csv')




# In[ ]:




