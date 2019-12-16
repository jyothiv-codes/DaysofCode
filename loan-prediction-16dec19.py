#!/usr/bin/env python
# coding: utf-8

# In[199]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 


# In[200]:


df = pd.read_csv("train.csv")
print(df.head(3))


# In[201]:


df.columns


# In[202]:


print(df.isnull().sum())


# In[203]:


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


# In[204]:


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


# In[205]:


df.loc[df.Dependents=="3+","Dependents"]=3
print(df['Dependents'].unique())
d={}



# In[206]:


for i in range(len(df['Dependents'])):
    #print(i)
    x=df['Dependents'].iloc[i]
    d[x]=d.get(x,0)+1
    #print(d[i])
print(d)


# In[207]:


df['Dependents']=df['Dependents'].fillna(0)


# In[208]:


df['Self_Employed'].unique()


# In[209]:


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


# In[210]:


df['Self_Employed'].unique()


# In[211]:


df.columns


# In[212]:


df.isnull().sum()


# In[213]:


df['LoanAmount'].unique()


# In[214]:


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


# In[215]:


df['LoanAmount']=df['LoanAmount'].fillna(m)


# In[216]:


df['LoanAmount'].unique()


# In[217]:



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


# In[218]:


df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(m)


# In[219]:



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


# In[220]:


df.isnull().sum()


# In[221]:


from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as ss
import itertools


# In[222]:


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


# In[223]:


cols = ["Gender", "Married", "Education","Self_Employed","Loan_Status","Property_Area"]
corrM = np.zeros((len(cols),len(cols)))
import itertools
# there's probably a nice pandas way to do this
for col1, col2 in itertools.combinations(cols, 2):
    idx1, idx2 = cols.index(col1), cols.index(col2)
    corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
    corrM[idx2, idx1] = corrM[idx1, idx2]


# In[224]:


corr = pd.DataFrame(corrM, index=cols, columns=cols)
fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");


# In[ ]:





# In[225]:


df.loc[df.Loan_Status=="N","Loan_Status"]="1"
df.loc[df.Loan_Status=="Y","Loan_Status"]="0"


# In[ ]:





# In[ ]:





# In[ ]:





# In[226]:


df.columns


# In[227]:


import pandas as pd

one_hot = pd.get_dummies(df['Property_Area'])
df = df.drop('Property_Area',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[228]:


df.columns


# In[229]:


df=df.drop(['Gender','Self_Employed'],axis=1)


# In[230]:


import pandas as pd

one_hot = pd.get_dummies(df['Self_Employed'])
df = df.drop('Self_Employed',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[231]:


df.columns


# In[ ]:





# In[232]:


import pandas as pd

one_hot = pd.get_dummies(df['Education'])
df = df.drop('Education',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[233]:


df.loc[df.Married=="No","Married"]="1m"
df.loc[df.Married=="Yes","Married"]="0m"


# In[234]:


import pandas as pd

one_hot = pd.get_dummies(df['Married'])
df = df.drop('Married',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[235]:


import pandas as pd

one_hot = pd.get_dummies(df['Gender'])
df = df.drop('Gender',axis = 1)
df = df.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[ ]:





# In[236]:


df.columns


# In[237]:


df.head(3)


# In[238]:


test=pd.read_csv("test_lAUu6dG.csv")


# In[239]:


test.isnull().sum()


# In[240]:


test=test.drop(['Gender','Self_Employed'],axis=1)


# In[241]:


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


# In[242]:


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



# In[243]:


test.loc[test.Dependents=="3+","Dependents"]=3
print(test['Dependents'].unique())
d={}


for i in range(len(test['Dependents'])):
    #print(i)
    x=test['Dependents'].iloc[i]
    d[x]=d.get(x,0)+1
    #print(d[i])
print(d)





# In[244]:



test['Dependents']=test['Dependents'].fillna(0)


# In[245]:


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



# In[246]:


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


# In[247]:



test['LoanAmount']=test['LoanAmount'].fillna(m)


# In[248]:


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




# In[249]:


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


# In[250]:


df.dtypes


# In[ ]:





# In[251]:


x_train = df.drop('Loan_Status', axis=1)
x_train=x_train.drop('Loan_ID',axis=1)
y_train = df['Loan_Status']


# In[252]:


import pandas as pd
from sklearn import preprocessing

x = x_train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_train = pd.DataFrame(x_scaled)


# In[168]:


from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# In[ ]:





# In[ ]:





# In[169]:


model = GaussianNB()
model.fit(x_train,y_train)
# make predictions
print(model)


# In[253]:


###After dropping columns based on correlation matrix


"""from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,r2_score,mean_absolute_error

model = LogisticRegression()
model.fit(x_train, y_train)
 """


# In[254]:


submission=pd.read_csv("sample_submission_49d68Cx.csv")


# In[255]:


copy=test


# In[256]:


dfans=copy['Loan_ID']


# In[257]:


copy.head(3)


# In[258]:


print(test.isnull().sum())


# In[259]:


test=test.drop('Loan_ID',axis=1)


# In[260]:


import pandas as pd

one_hot = pd.get_dummies(test['Property_Area'])
test = test.drop('Property_Area',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[261]:


import pandas as pd

one_hot = pd.get_dummies(test['Self_Employed'])
test = test.drop('Self_Employed',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[262]:


import pandas as pd

one_hot = pd.get_dummies(test['Education'])
test = test.drop('Education',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[263]:


test.loc[test.Married=="No","Married"]="1m"
test.loc[test.Married=="Yes","Married"]="0m"


# In[264]:


import pandas as pd

one_hot = pd.get_dummies(test['Married'])
test = test.drop('Married',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[265]:


import pandas as pd

one_hot = pd.get_dummies(test['Gender'])
test = test.drop('Gender',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


# In[266]:


test.loc[test.Dependents=="3+","Dependents"]="3"


# In[267]:


predicted = model.predict(test)


# In[268]:


print(predicted)


# In[269]:


predicted.shape


# In[270]:


test.shape


# In[271]:


submission.shape


# In[272]:


submission=submission.iloc[0:0]


# In[273]:


submission.shape


# In[274]:


print(submission)


# In[275]:


#submission['Loan_Status'].replace(0, 'N',inplace=True)
#submission['Loan_Status'].replace(1, 'Y',inplace=True)
print(dfans)


# In[276]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')


# In[277]:


print(copy.head(3))


# In[278]:


submission['Loan_ID']=dfans


# In[279]:


submission['Loan_Status']=predicted 


# In[280]:


submission.loc[submission.Loan_Status=='0',"Loan_Status"]="N"
submission.loc[submission.Loan_Status=='1',"Loan_Status"]="Y"


# In[281]:


submission['Loan_Status']


# In[282]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistics.csv')


# In[ ]:




