
# coding: utf-8

# In[18]:


import pandas as pd
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[19]:


df = pd.read_csv("textTrainData.txt", sep="\t", encoding='latin1')


# In[20]:


tf = pd.read_csv("textTestData.txt", sep="\t", encoding="latin1")


# In[21]:


df.head()


# In[22]:


tf.head()


# In[23]:


#TF-IDF Vectorizer
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')


# In[24]:


y_train = df.Sentiment


# In[25]:


y_test = tf.Sentiment


# In[26]:


x_train = vectorizer.fit_transform(df.Sentence)


# In[27]:


x_test = vectorizer.transform(tf.Sentence)


# In[28]:


print (y_train.shape)
print (x_train.shape)


# In[29]:


print(y_test.shape)
print(x_test.shape)


# In[30]:


#Train the Naive Bayes classifier
clf = naive_bayes.MultinomialNB()
clf.fit(x_train, y_train)


# In[31]:


predictions = clf.predict(x_test)
predictions


# In[32]:


print( "Train Accuracy is :", accuracy_score(y_train, clf.predict(x_train)))


# In[33]:


print( "Test Accuracy is :", accuracy_score(y_test, clf.predict(x_test)))


# In[34]:


print( " Confusion matrix for train:")
print(confusion_matrix(y_train, clf.predict(x_train)))
print( " Confusion matrix for test :")
print(confusion_matrix(y_test, clf.predict(x_test)))

