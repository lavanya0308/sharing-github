#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all the lib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[3]:


# read the dataset using pandas
data = pd.read_csv('C://Users//lmohan2//Desktop//New folder//test//Salary_Data.csv')
data.head()


# In[4]:


# Provides some information regarding the columns in the data
data.info()


# In[5]:


# this describes the basic stat behind the dataset used 
data.describe()


# In[6]:


# These Plots help to explain the values and how they are scattered

plt.figure(figsize=(12,6))
sns.pairplot(data,x_vars=['YearsExperience'],y_vars=['Salary'],size=7,kind='scatter')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.show()


# In[7]:


# Cooking the data
X = data['YearsExperience']
X.head()


# In[8]:


# Cooking the data
y = data['Salary']
y.head()


# In[19]:


# Import Segregating data from scikit learn
from sklearn.model_selection import train_test_split
# Split the data for train and test 
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)
# Create new axis for x column
X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]
#we need to convert this to 2d array


# In[22]:


# Importing Linear Regression model from scikit learn
from sklearn.linear_model import LinearRegression
# Fitting the model
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[30]:


# Predicting the Salary for the Train values
y_predicted=lr.predict(X_train)
# Plotting the actual and predicted values
c = [i for i in range (1,len(y_train)+1,1)]
plt.plot(c,y_train,color='r',linestyle='-')
plt.plot(c,y_predicted,color='b',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()


# In[23]:


# Predicting the Salary for the Test values
y_pred = lr.predict(X_test)


# In[24]:


# Plotting the actual and predicted values for the test data

c = [i for i in range (1,len(y_test)+1,1)]
plt.plot(c,y_test,color='r',linestyle='-')
plt.plot(c,y_pred,color='b',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()


# In[31]:


# plotting the error
c = [i for i in range(1,len(y_train)+1,1)]
plt.plot(c,y_train-y_predicted,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()


# In[43]:


# Importing metrics for the evaluation of the model
from sklearn.metrics import r2_score,mean_squared_error
# Calculate R square vale
rsq = r2_score(y_train,y_predicted)
#print('mean squared error:',mse) 
print('R SQUARE:',rsq) 


# In[42]:


#RMSE
from math import sqrt
rms_d = sqrt(mean_squared_error(y_train,y_predicted))
print('RMSE:',rms_d) 


# In[44]:


#Mean absolute Error

from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(y_train,y_predicted))


# In[45]:


#Mean square Error
from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(y_train,y_predicted))


# In[46]:


# Intecept and coeff of the line
print('Intercept of the model:',lr.intercept_)
print('Coefficient of the line:',lr.coef_)

#y = 25202.8 + 9731.2x


# In[ ]:




