#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #importing function
import pandas as pd
col_names=["sepal_length","sepal_width","petal_length","petal_width","type"]
data=pd.read_csv("iris_dataset.csv",skiprows=1,header=None,names=col_names) #taking values from excel sheet
data.head(10)
X=data.iloc[:,:-1].values
Y1=data.iloc[:,-1].values
Y=Y1.reshape(-1,1)
from  sklearn.model_selection import train_test_split # importing 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=41) #spliting the values among X & Y
from sklearn import tree #
classifier=tree.DecisionTreeClassifier(min_samples_split=3,max_depth=3,criterion="entropy")
classifier.fit(X_train,Y_train)
classifier.score(X_test,Y_test)
tree.plot_tree(classifier) #ploting as a tree


# In[ ]:




