#!/usr/bin/env python
# coding: utf-8

# In[97]:


import tensorflow as tf


# In[141]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[142]:


# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,-1]


# In[143]:


# Feature Scaling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[144]:


# Part 2 - Now let's make the ANN!


# In[145]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU
from tensorflow.keras.layers import Dropout


# In[146]:


# Initialising the ANN
classifier = Sequential()


# In[147]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(units=8,activation='relu'))


# In[148]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6,activation='relu'))


# In[149]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(units=1,activation='relu'))


# In[150]:


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[151]:


model_history=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=50)


# In[152]:


# list all data in history

print(model_history.history.keys())


# In[153]:



# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[154]:


# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[155]:


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[156]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[157]:


# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)


# In[158]:


score


# In[159]:



sample_input = np.array([[2, 90, 80, 25, 50, 25, 0.25, 30]]) 

scaled_input = sc.transform(sample_input)



prediction = classifier.predict(scaled_input)
prediction = (prediction > 0.5).astype('int32')

print(prediction)


# In[ ]:





# In[ ]:




