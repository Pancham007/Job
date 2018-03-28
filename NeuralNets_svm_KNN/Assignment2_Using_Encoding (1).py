
# coding: utf-8

# In[96]:


# We will start with the breast cancer study again ...
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools


# In[97]:


myDataframe1 = pd.read_csv('adultTrain.data', names = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
myDataframe2 =pd.read_csv('adultTest.data', names = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])


# In[98]:


classes=myDataframe1.iloc[:,-1].unique()


# In[99]:


myDataframe1 = myDataframe1.replace(' ?',np.NaN)
myDataframe1 = myDataframe1.dropna()
#myDataframe1.drop(myDataframe1.columns[[2, 4, 5, 7, 9, 10, 11, 12]], axis=1, inplace=True)
le = preprocessing.LabelEncoder()
myDataframe1 = myDataframe1.apply(le.fit_transform)
myDataframe1


# In[100]:


myDataframe2 = myDataframe2.replace(' ?',np.NaN)
myDataframe2 = myDataframe2.dropna()
#myDataframe2.drop(myDataframe2.columns[[2, 4, 5, 7, 9, 10, 11, 12]], axis=1, inplace=True)
le = preprocessing.LabelEncoder()
myDataframe2 = myDataframe2.apply(le.fit_transform)


# In[101]:


#print(myDataframe.iloc[:,7].value_counts())
X_train = myDataframe1.iloc[:,:-1]
y_train = myDataframe1.iloc[:,-1]


# In[102]:


X_test = myDataframe2.iloc[:,:-1]
y_test = myDataframe2.iloc[:,-1]


# In[103]:


#X_train, X_test, y_train, y_test = train_test_split(data, y, random_state = 0, test_size=0.25)


# In[104]:


X_train.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

# In[105]:


# We'll build the network next. Let's do the same architecure as before, three
# hidden layers of the same size as the input data ...
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)


# In[106]:


predictions = mlp.predict(X_test)
#y_test


# In[107]:


import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[108]:


cm = confusion_matrix(y_test,predictions)


# In[109]:


plot_confusion_matrix(cm,classes)


# In[110]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, mlp.predict(X_test))


# In[111]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=1)


# In[112]:


KNN.fit(X_train,y_train)


# In[113]:


predictions = KNN.predict(X_test)


# In[114]:


cm = confusion_matrix(y_test,predictions)


# In[115]:


plot_confusion_matrix(cm,classes)


# In[116]:


accuracy_score(y_test, KNN.predict(X_test))


# In[117]:


from sklearn import svm


# In[118]:


clf = svm.SVC()


# In[119]:


clf.fit(X_train, y_train)  


# In[120]:


predictions = clf.predict(X_test)


# In[121]:


cm = confusion_matrix(y_test,predictions)


# In[122]:


accuracy_score(y_test, clf.predict(X_test))


# In[123]:


plot_confusion_matrix(cm,classes)

