#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# In[3]:


dataset = pd.read_csv('spam.csv',encoding = "ISO-8859-1")


# In[4]:


dataset.head()


# In[5]:


dataset.describe()


# In[6]:


print("Email spam detection dataset is: \n",dataset)


# In[18]:


ps=PorterStemmer()
lemmatize=WordNetLemmatizer()
corpus=[]
for i in range(0,len(df)):
  review=re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])
  review = review.lower()
  review = review.split()
    
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y=pd.get_dummies(dataset['v1'])
y=y.iloc[:,1].values


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)  


# In[21]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)


# In[22]:


y_pred=spam_detect_model.predict(X_test)


# In[23]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))


# In[24]:


print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))


# In[25]:


print("Classification report: {}".format(classification_report(y_test,y_pred)))


# In[ ]:




