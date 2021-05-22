#!/usr/bin/env python
# coding: utf-8

# In[102]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC




# lIris Dataset Tutorial
iris = datasets.load_iris()
iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target

iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.dropna(how="all", inplace=True)

iris_df.head()


# In[103]:


X_train, X_test, y_train, y_test = train_test_split(iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']], iris_df['class'], random_state = 0)

print(X_train.shape)
print(X_test.shape)


# In[104]:


knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)


# In[105]:


prediction = knn.predict(X_train)
knn.score(X_test, y_test)


# In[106]:


classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_pred,y_test)
print(accuracy)


# In[107]:


classifier = ComplementNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)


# In[108]:


classifier = BernoulliNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)


# In[109]:


classifier = DecisionTreeClassifier(max_depth = 8, random_state=3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_pred,y_test)
print(accuracy)


# In[110]:


model = SVC(gamma='auto')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print(accuracy)


# In[91]:


classifier = LinearDiscriminantAnalysis()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_pred,y_test)
print(accuracy)


# In[ ]:





# In[ ]:




