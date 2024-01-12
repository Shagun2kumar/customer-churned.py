#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# In[2]:


df=pd.read_csv('customer-churn.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


df.dtypes


# In[9]:


df.duplicated().any()


# In[10]:


df['TotalCharges'].dtype


# In[11]:


df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges'].dtype


# In[12]:


categories=[ 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']
num_features=["tenure","MonthlyCharges","TotalCharges"]
target="churn"


# In[13]:


df.skew(numeric_only=True)


# In[15]:


df[num_features].describe()


# In[16]:


features='Contract'
fig,ax=plt.subplots(1,2,figsize=(12,4))
df[df.Churn=="No"][features].value_counts().plot(kind='bar',ax=ax[0]).set_title('not churned')
df[df.Churn=="Yes"][features].value_counts().plot(kind='bar',ax=ax[1]).set_title('churned')


# In[17]:


df.drop(['customerID'],axis=1,inplace=True)


# In[18]:


df1=pd.get_dummies(data=df,columns=[ 'gender',  'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'Churn'],drop_first=True)


# In[19]:


df1.head()


# In[20]:


df1.shape


# In[21]:


imputer=SimpleImputer(missing_values=np.nan, strategy="mean")
df1.TotalCharges=imputer.fit_transform(df1["TotalCharges"].values.reshape(-1,1))


# In[22]:


scaler=StandardScaler()
scaler.fit(df1.drop(['Churn_Yes'],axis=1))
scaled_features=scaler.transform(df1.drop('Churn_Yes',axis=1))


# In[23]:


x=scaled_features
y=df1['Churn_Yes']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=44)


# In[24]:


model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred_model=model.predict(x_test)


# In[25]:


print(classification_report(y_test,y_pred_model))


# In[26]:


matrix_model=confusion_matrix(y_test,y_pred_model)


# In[28]:


plt.matshow(matrix_model)
plt.xlabel('predicted class ')
plt.ylabel('actual class')
for i in range(2):
    for j in range(2):
        plt.text(j,i,matrix_model[i,j],ha='center',va='center')
plt.xticks([0,1],["Not churned","churned"])
plt.yticks([0,1],["Not churned","churned"])
plt.show()


# In[29]:


model.score(x_train,y_train)


# In[30]:


accuracy_score(y_test,y_pred_model)


# In[ ]:




