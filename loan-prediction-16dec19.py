#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 


# In[104]:


df = pd.read_csv("train.csv")
print(df.head(3))


# In[105]:


df.columns


# In[106]:


print(df.isnull().sum())



male=0
female=0
for i in range(len(df['Gender'])):
    if df['Gender'].iloc[i]=="Male":
        male+=1
    else:
        female+=1
print(male,female)    
if male>female:
    df['Gender']=df['Gender'].fillna("Male")
else:
    df['Gender']=df['Gender'].fillna("Female")


# In[108]:


married=0
nmarried=0
for i in range(len(df['Married'])):
    if df['Married'].iloc[i]=="Yes":
        married+=1
    else:
        nmarried+=1
print(married,nmarried)    
if married>nmarried:
    df['Married']=df['Married'].fillna("Yes")
else:
    df['Married']=df['Married'].fillna("No")


# In[109]:


df.loc[df.Dependents=="3+","Dependents"]=3
print(df['Dependents'].unique())
d={}



# In[110]:


for i in range(len(df['Dependents'])):
    #print(i)
    x=df['Dependents'].iloc[i]
    d[x]=d.get(x,0)+1
    #print(d[i])
print(d)


# In[111]:


df['Dependents']=df['Dependents'].fillna(0)


# In[112]:


df['Self_Employed'].unique()


# In[113]:


self=0
nself=0
for i in range(len(df['Self_Employed'])):
    if df['Self_Employed'].iloc[i]=="Yes":
        self+=1
    else:
        nself+=1
print(self,nself)    
if self>nself:
    df['Self_Employed']=df['Self_Employed'].fillna("Yes")
else:
    df['Self_Employed']=df['Self_Employed'].fillna("No")


# In[114]:


df['Self_Employed'].unique()


# In[115]:


df.columns


# In[116]:


df.isnull().sum()
df['LoanAmount'].unique()

from statistics import mean
m=0
s=0
n=0
for i in range(len(df['LoanAmount'])):
    #print(df['LoanAmount'].iloc[i],type(df['LoanAmount'].iloc[i]))
    if df['LoanAmount'].iloc[i]>0:
        #print("HI")
        s+=df['LoanAmount'].iloc[i]
        n+=1
m=s/n
print(m)


# In[119]:


df['LoanAmount']=df['LoanAmount'].fillna(m)


# In[120]:


df['LoanAmount'].unique()


# In[121]:



from statistics import mean
m=0
s=0
n=0
for i in range(len(df['Loan_Amount_Term'])):
    #print(df['LoanAmount'].iloc[i],type(df['LoanAmount'].iloc[i]))
    if df['Loan_Amount_Term'].iloc[i]>0:
        #print("HI")
        s+=df['Loan_Amount_Term'].iloc[i]
        n+=1
m=s/n
print(m)


# In[122]:


df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(m)


# In[123]:



from statistics import mean
m=0
s=0
n=0
for i in range(len(df['Credit_History'])):
    #print(df['LoanAmount'].iloc[i],type(df['LoanAmount'].iloc[i]))
    if df['Credit_History'].iloc[i]>0:
        #print("HI")
        s+=df['Credit_History'].iloc[i]
        n+=1
m=s/n
print(m)
df['Credit_History']=df['Credit_History'].fillna(m)


# In[124]:


df.isnull().sum()


# In[125]:


df.loc[df.Loan_Status=="N","Loan_Status"]="1"
df.loc[df.Loan_Status=="Y","Loan_Status"]="0"


# In[126]:


df.columns


# In[127]:


import pandas as pd

one_hot = pd.get_dummies(df['Property_Area'])
df = df.drop('Property_Area',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))
df.columns

import pandas as pd

one_hot = pd.get_dummies(df['Self_Employed'])
df = df.drop('Self_Employed',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[130]:


df.columns


# In[ ]:





# In[131]:


import pandas as pd

one_hot = pd.get_dummies(df['Education'])
df = df.drop('Education',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[132]:


df.loc[df.Married=="No","Married"]="1m"
df.loc[df.Married=="Yes","Married"]="0m"


# In[133]:


import pandas as pd

one_hot = pd.get_dummies(df['Married'])
df = df.drop('Married',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[134]:


import pandas as pd

one_hot = pd.get_dummies(df['Gender'])
df = df.drop('Gender',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[ ]:





# In[135]:


df.columns


# In[136]:


df.head(3)


# In[190]:


test=pd.read_csv("test_lAUu6dG.csv")


# In[191]:


test.isnull().sum()


# In[192]:


#test=test.dropna()
male=0
female=0
for i in range(len(test['Gender'])):
    if test['Gender'].iloc[i]=="Male":
        male+=1
    else:
        female+=1
print(male,female)    
if male>female:
    test['Gender']=test['Gender'].fillna("Male")
else:
    test['Gender']=test['Gender'].fillna("Female")


# In[193]:


married=0
nmarried=0
for i in range(len(test['Married'])):
    if test['Married'].iloc[i]=="Yes":
        married+=1
    else:
        nmarried+=1
print(married,nmarried)    
if married>nmarried:
    test['Married']=test['Married'].fillna("Yes")
else:
    test['Married']=test['Married'].fillna("No")



# In[194]:


test.loc[test.Dependents=="3+","Dependents"]=3
print(test['Dependents'].unique())
d={}


for i in range(len(test['Dependents'])):
    #print(i)
    x=test['Dependents'].iloc[i]
    d[x]=d.get(x,0)+1
    #print(d[i])
print(d)





# In[195]:



test['Dependents']=test['Dependents'].fillna(0)


# In[196]:


self=0
nself=0
for i in range(len(test['Self_Employed'])):
    if test['Self_Employed'].iloc[i]=="Yes":
        self+=1
    else:
        nself+=1
print(self,nself)    
if self>nself:
    test['Self_Employed']=test['Self_Employed'].fillna("Yes")
else:
    test['Self_Employed']=test['Self_Employed'].fillna("No")



# In[197]:


from statistics import mean
m=0
s=0
n=0
for i in range(len(test['LoanAmount'])):
    #print(df['LoanAmount'].iloc[i],type(df['LoanAmount'].iloc[i]))
    if test['LoanAmount'].iloc[i]>0:
        #print("HI")
        s+=test['LoanAmount'].iloc[i]
        n+=1
m=s/n
print(m)


# In[198]:



test['LoanAmount']=test['LoanAmount'].fillna(m)


# In[199]:


from statistics import mean
m=0
s=0
n=0
for i in range(len(test['Loan_Amount_Term'])):
    #print(df['LoanAmount'].iloc[i],type(df['LoanAmount'].iloc[i]))
    if test['Loan_Amount_Term'].iloc[i]>0:
        #print("HI")
        s+=test['Loan_Amount_Term'].iloc[i]
        n+=1
m=s/n
print(m)


test['Loan_Amount_Term']=test['Loan_Amount_Term'].fillna(m)




# In[200]:


from statistics import mean
m=0
s=0
n=0
for i in range(len(test['Credit_History'])):
    #print(df['LoanAmount'].iloc[i],type(df['LoanAmount'].iloc[i]))
    if test['Credit_History'].iloc[i]>0:
        #print("HI")
        s+=test['Credit_History'].iloc[i]
        n+=1
m=s/n
print(m)
test['Credit_History']=test['Credit_History'].fillna(m)


# In[ ]:





# In[ ]:





# In[201]:


from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# In[ ]:





# In[202]:


x_train = df.drop('Loan_Status', axis=1)
x_train=x_train.drop('Loan_ID',axis=1)
y_train = df['Loan_Status']


# In[203]:


model = GaussianNB()
model.fit(x_train,y_train)
# make predictions
print(model)


# In[204]:


submission=pd.read_csv("sample_submission_49d68Cx.csv")


# In[ ]:


submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']


# In[ ]:


submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[ ]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')


# In[205]:


copy=test


# In[206]:


dfans=copy['Loan_ID']


# In[207]:


copy.head(3)


# In[208]:


print(test.isnull().sum())


# In[217]:


test=test.drop('Loan_ID',axis=1)


# In[209]:


import pandas as pd

one_hot = pd.get_dummies(test['Property_Area'])
test = test.drop('Property_Area',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[210]:


import pandas as pd

one_hot = pd.get_dummies(test['Self_Employed'])
test = test.drop('Self_Employed',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[211]:


import pandas as pd

one_hot = pd.get_dummies(test['Education'])
test = test.drop('Education',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[212]:


test.loc[test.Married=="No","Married"]="1m"
test.loc[test.Married=="Yes","Married"]="0m"


# In[213]:


import pandas as pd

one_hot = pd.get_dummies(test['Married'])
test = test.drop('Married',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[214]:


import pandas as pd

one_hot = pd.get_dummies(test['Gender'])
test = test.drop('Gender',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[215]:


test.loc[test.Dependents=="3+","Dependents"]="3"


# In[218]:


predicted = model.predict(test)


# In[219]:


print(predicted)


# In[220]:


predicted.shape


# In[221]:


test.shape


# In[222]:


submission.shape


# In[223]:


submission=submission.iloc[0:0]



submission.shape
print(submission)
print(dfans)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')
print(copy.head(3))
submission['Loan_ID']=dfans

submission['Loan_Status']=predicted 

submission.loc[submission.Loan_Status=='0',"Loan_Status"]="N"
submission.loc[submission.Loan_Status=='1',"Loan_Status"]="Y"





submission['Loan_Status']=submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status']=submission['Loan_Status'].replace(1, 'Y',inplace=True)



submission['Loan_Status']


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistics.csv')




