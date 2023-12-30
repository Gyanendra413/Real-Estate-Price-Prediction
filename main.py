#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


HouseDF = pd.read_csv('house price.csv')


# In[4]:


#To see the head of the dataset
HouseDF.head()


# In[5]:


#Getting info of the dataset
HouseDF.info()


# In[6]:


#To describe the dataframe(get deviations)
HouseDF.describe()


# In[ ]:





# In[7]:


#To get the columns of the dataframe
HouseDF.columns


# In[8]:


#Getting plots of the various parameters of the dataset
sns.pairplot(HouseDF)


# In[9]:


#To get the correlation of the data frame(Heat Map)
#corr=>correlation
#annot=>annotation
sns.heatmap(HouseDF.corr(), annot=True)


# In[10]:


#Initializing the independent variables
X = HouseDF[['House Area']]
y = HouseDF['Price']


# In[11]:


#Creating training and testing dataset


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


#Creating Train set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


lm = LinearRegression()


# In[16]:


#Putting the train set in Linear Regression model
lm.fit(X_train, y_train)


# In[17]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])


# In[18]:


coeff_df


# In[19]:


#Predicting using test dataset
predictions = lm.predict(X_test)


# In[20]:


#Plotting the predictions against y_test
plt.scatter(y_test,predictions)


# In[21]:


sns.histplot((y_test-predictions),bins=50);


# In[22]:


#Bellshaped histogram reveals that the models are well predicted.


# In[23]:


# define input
new_input = [[36000]]
new_output = lm.predict(new_input)
print("Given Area is ",new_input)
print("Predicted price is ",new_output)
accuracy = lm.score(X_test, y_test)
print('Accuracy : ',accuracy*100,'%')


# In[26]:


# Array of predicted results 
predictions


# In[27]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:
# Pandey ka comment line. Ye conflict dega...



