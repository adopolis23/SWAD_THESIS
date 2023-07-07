#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# the four different states of the XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])


# In[62]:


weights1 = model.get_weights()


# In[63]:


model.fit(training_data, target_data, epochs=10, verbose=0)


# In[64]:


weights2 = model.get_weights()


# In[65]:


def addWeights(w1, w2):
    tmp = [w1, w2]
    new = list()
    for wt in zip(*tmp):
        new.append(
            np.array([np.array(w).sum(axis=0) for w in zip(*wt)])
        )
    return new


# In[66]:


new_w = addWeights(weights1, weights2)
new_w


# In[67]:


def divideBy(w1, n):
    sol = list()
    
    for x in w1:
        if len(x.shape) == 2:
            sol.append(
                np.array([[y/n for y in z] for z in x])
            )
        if len(x.shape) == 1:
            sol.append(
                np.array([z/n for z in x])
            )
    
    return sol
    
    
            
new2 = divideBy(new_w, 2)
new2


# In[ ]:




