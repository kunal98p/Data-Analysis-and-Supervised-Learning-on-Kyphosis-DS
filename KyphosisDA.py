
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('kyphosis.csv')


# In[3]:


df.head()


# In[4]:


sns.pairplot(df,hue='Kyphosis',palette='Set1')


# DECISION TREE IMPLEMENTATION

# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[7]:


from sklearn.tree import DecisionTreeClassifier


# In[8]:


dtree = DecisionTreeClassifier()


# In[9]:


dtree.fit(X_train,y_train)


# In[10]:


predictions = dtree.predict(X_test)


# In[11]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# RANDOM FORESTIMPLEMENTATION

# In[13]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[14]:


rfc_pred = rfc.predict(X_test)


# In[15]:


print(confusion_matrix(y_test,rfc_pred))


# In[16]:


print(classification_report(y_test,rfc_pred))


# In[ ]:


#Comparison shows the Random Forest Classifier perform better than the Decision Tree Classifier

