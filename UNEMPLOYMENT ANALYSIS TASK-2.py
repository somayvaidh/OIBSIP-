#!/usr/bin/env python
# coding: utf-8

# IMPORTING LIBRARIES

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[49]:


df=pd.read_csv("Unemployment in India.csv")
df1=pd.read_csv("Unemployment_Rate_upto_11_2020.csv")


# In[50]:


df.head()


# In[51]:


df1.head()


# In[52]:


df.info()


# In[53]:


df1.info()


# In[54]:


df.shape


# In[55]:


df1.shape


# In[56]:


df.describe


# In[57]:


df1.describe


# In[58]:


#missing data


# In[59]:


df.isnull().sum() #returning the count of values in each column


# In[60]:


df1.isnull().sum()


# In[61]:


#dropping null values
df=df.dropna()


# In[62]:


df.isnull().sum()


# In[63]:


df.shape


# In[64]:


df.duplicated().sum()


# In[65]:


df1.duplicated().sum()


# In[66]:


df.columns


# In[67]:


df1.columns


# In[68]:


df.head()


# In[69]:


df['Region'].value_counts().idxmax()


# In[70]:


df1['Region'].value_counts().idxmax()


# In[71]:


df['Region'].value_counts().idxmin()


# In[72]:


df1['Region'].value_counts().idxmin()


# In[82]:


sns.pairplot(df)


# In[83]:


sns.pairplot(df1)


# In[87]:


fig=plt.figure(figsize=(5,5))
sns.histplot(x=' Estimated Labour Participation Rate (%)', data=df, kde=True, hue='Area')
plt.title('Labour Participation according to Area')
plt.xlabel('Labour Participation Rate')


# In[88]:


fig=plt.figure(figsize=(5,5))
sns.histplot(x=' Estimated Unemployment Rate (%)', data=df, kde=True, hue='Area')
plt.title('Unemployment according to Area')
plt.xlabel('Unemployment Rate')
plt.show()


# In[89]:


fig = plt.figure(figsize = (5, 5))
sns.lineplot(y=' Estimated Unemployment Rate (%)', x=' Date', data=df)
plt.title('Unemployment according to Date')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel('Unemployment Rate')
plt.show()


# In[100]:


fig = plt.figure(figsize = (7, 7))
sns.lineplot(y=' Estimated Labour Participation Rate (%)', x=' Date', data=df)
plt.title('Labour Participation according to Date')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel('Labour Participation Rate')
plt.show()


# In[94]:


y=df[' Estimated Unemployment Rate (%)']
x=df['Region']
plt_1 = plt.figure(figsize=(10, 10))
plt.title('Umemployment Rate', fontweight='bold' ,fontsize=20)
plt.xlabel("States",fontweight='bold',fontsize=20)
plt.ylabel("Estimated Unemployment rate",fontweight='bold',fontsize=20)
plt.xticks(rotation='vertical',fontsize=12)
sns.histplot(x, color='lavender')


# In[98]:


fig = plt.figure(figsize = (5, 5))
plt.scatter(df1['Region.1'], df1[' Estimated Labour Participation Rate (%)'])
plt.title('Labour Participation according to Region')
plt.xlabel('Region')
plt.ylabel('Labour Participation Rate')
plt.show()


# In[97]:


fig = plt.figure(figsize = (5, 5))
plt.scatter(df1[' Date'], df1[' Estimated Employed'])

plt.title('Unemployment according to Region')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate')
plt.show()


# In[101]:


#Now let’s have a look at the correlation between the features of this dataset:
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr())
plt.show()


# In[102]:


#Now let’s have a look at the correlation between the features of this dataset:
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(df1.corr())
plt.show()


# In[104]:


df1.columns


# In[110]:


plt.figure(figsize=(12,12))
plt.title("Indian Unemplyment")
sns.histplot(x=" Estimated Employed", hue="Region.1",data=df1)
plt.show()


# In[114]:


plt.figure(figsize=(12, 11))
plt.title("Indian Unemployment")
sns.histplot(x=" Estimated Unemployment Rate (%)", hue="Region.1", data=df1)
plt.show()


# In[116]:


plt.figure(figsize=(12, 11))
plt.title("Indian Unemployment")
sns.histplot(x=" Estimated Unemployment Rate (%)", hue=" Estimated Labour Participation Rate (%)", data=df1)
plt.show()


# In[124]:


import plotly.express as px
unemploment = df1[["Region", "Region.1", " Estimated Unemployment Rate (%)"]]
figure = px.sunburst(unemploment, path=["Region.1", "Region"], 
                     values=" Estimated Unemployment Rate (%)", 
                     width=700, height=700, color_continuous_scale="RdY1Gn", 
                     title="Unemployment Rate in India")
figure.show()


# In[ ]:




