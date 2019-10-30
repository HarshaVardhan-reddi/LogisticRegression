#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[2]:


dataset = pd.read_csv('train.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.index


# In[5]:


print('total no of passengers : ', len(dataset.index))


# # Data analysis

# In[6]:


sns.countplot(x='Survived',data = dataset)


# In[7]:


sns.countplot(x='Survived',hue = 'Sex',data = dataset)


# In[8]:


sns.countplot(x='Survived',hue='Pclass',data = dataset)


# In[9]:


dataset['Age'].plot.hist()
plt.xlabel('age')


# In[10]:


sns.countplot(x='SibSp',data=dataset)


# # Data wrangling

# In[11]:


dataset.isnull()


# In[12]:


dataset.isnull().sum()


# In[13]:


sns.heatmap(dataset.isnull(),yticklabels=False)


# In[14]:


sns.boxplot(x = 'Pclass',y ='Age',data = dataset)


# In[15]:


dataset.head(2)


# In[16]:


dataset.drop(['Cabin'],inplace=True,axis=1)


# In[17]:


dataset.head(2)


# In[18]:


dataset.dropna(inplace=True)


# In[19]:


sns.heatmap(dataset.isnull(),yticklabels=False)


# In[20]:


sex = pd.get_dummies(dataset['Sex'])
sex.head(2)


# In[21]:


sex = pd.get_dummies(dataset['Sex'],drop_first = True)
sex.head(2)


# In[22]:


embark = pd.get_dummies(dataset['Embarked'])
embark.head(2)


# In[23]:


embark = pd.get_dummies(dataset['Embarked'],drop_first = True)
embark.head(2)


# In[24]:


pclass = pd.get_dummies(dataset['Pclass'])
pclass.head()


# In[25]:


pclass = pd.get_dummies(dataset['Pclass'],drop_first = True)
pclass.head()


# In[26]:


dataset_t = pd.concat([dataset,sex,embark,pclass],axis = 1)
dataset_t.head(2)


# In[27]:


dataset_t.drop(['PassengerId','Pclass','Name','Sex','Embarked','Ticket'],axis=1,inplace=True)


# # Train

# In[28]:


x = dataset_t.drop('Survived',axis=1)
y = dataset_t['Survived']


# In[29]:


from sklearn.linear_model import LogisticRegression


# In[30]:


logr = LogisticRegression()


# In[31]:


logr.fit(x,y)


# In[32]:


dataset_t.head(2)


# # testing

# In[38]:


data_test = pd.read_csv('test1.csv')
data_test.head(2)


# In[43]:


data_test.drop(data_test.columns[data_test.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
data_test.head(2)


# In[44]:


data_test.isnull()


# In[45]:


sns.heatmap(data_test.isnull(),yticklabels = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:


sex = pd.get_dummies(data_test['Sex'])
sex.head(4)


# In[47]:


sex = pd.get_dummies(data_test['Sex'],drop_first = True)
sex.head(4)


# In[48]:


pclass = pd.get_dummies(data_test['Pclass'],drop_first = True)
pclass.head()


# In[49]:


embark = pd.get_dummies(data_test['Embarked'],drop_first = True)
embark.head(2)


# In[52]:


testd = pd.concat([data_test,sex,pclass,embark],axis = 1)
testd.set_index('PassengerId',inplace = True)
testd.head(2)


# In[53]:


testd.drop(['Pclass','Name','Sex','Ticket','Embarked'],axis = 1,inplace = True)


# In[54]:


testd.head(2)


# In[60]:


x_in=testd.index


# In[57]:


y_predict = logr.predict(testd)
y_predict


# In[69]:


predictions = pd.DataFrame({'Survived':y_predict},index = x_in)


# In[70]:


predictions.head()


# In[71]:


predictions.to_csv('predictions.csv')





