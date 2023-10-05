#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


dataset=pd.read_csv("Advertising.csv")
dataset.head()


# In[33]:


dataset.columns


# In[34]:


dataset1 = dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
dataset


# In[60]:


dataset.size


# In[61]:


dataset.describe()


# In[35]:


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values


# In[62]:


dataset.shape


# In[36]:


print(X)


# In[37]:


print(Y)


# In[38]:


plt.figure(figsize=(9,9))
sns.distplot(dataset['Sales'])
plt.show()


# In[39]:


sns.pairplot(dataset)


# In[40]:


correlation=dataset.corr()
sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True)


# In[43]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[44]:


print(X_train)


# In[45]:


print(X_test)


# In[46]:


print(Y_train)


# In[47]:


print(Y_test)


# In[49]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)


# In[50]:


Y_pred=regressor.predict(X_test)


# In[51]:


print(Y_pred)


# In[52]:


coefficient = regressor.coef_
coefficient


# In[53]:


# calculating the intercept
intercept = regressor.intercept_
intercept


# In[56]:


# calculating the R squared value
from sklearn.metrics import r2_score
r2_score(Y_test, Y_pred)


# In[58]:


forecast=pd.DataFrame(data={'Forecasted Sales': Y_pred.flatten()})
forecast

