#!/usr/bin/env python
# coding: utf-8

#  ## Dragon Real Estate Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("data.csv")
housing.head()


# In[3]:


housing.info()


# In[4]:


housing['CHAS'].value_counts()


# In[5]:


housing.describe()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


housing.hist(bins=50,figsize=(20,15))


# In[9]:


import numpy as np
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(len(train_set))


# In[10]:


from sklearn.model_selection import StratifiedShuffleSplit
split= StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing['CHAS']):
    strat_train_set= housing.loc[train_index]
    strat_test_set= housing.loc[test_index]


# In[11]:


strat_train_set["CHAS"].value_counts()


# In[12]:


strat_test_set["CHAS"].value_counts()


# In[13]:


housing= strat_train_set.copy()


# In[14]:


corr_matrix= housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[15]:


from pandas.plotting import scatter_matrix
attributes= ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[16]:


housing.plot(kind='scatter',x='RM',y="MEDV",alpha=0.8)


# In[17]:


housing["TAXRM"]= housing["TAX"]/housing["RM"]


# In[18]:


housing.head()


# In[ ]:





# In[19]:


housing.describe()


# In[20]:


housing.plot(kind='scatter',x='TAXRM',y="MEDV",alpha=0.8)


# In[21]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels= strat_train_set["MEDV"].copy()


# In[22]:


from sklearn.impute import SimpleImputer
imputer= SimpleImputer(strategy="median")
imputer.fit(housing)


# In[23]:


imputer.statistics_


# In[24]:


X= imputer.transform(housing)


# In[25]:


X.shape


# In[26]:


housing_tr= pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()


# In[27]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline= Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler()),
])


# In[28]:


housing_num_tr= my_pipeline.fit_transform(housing)


# In[29]:


housing_num_tr.shape


# In[36]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[37]:


some_data= housing.iloc[:5]


# In[38]:


some_labels= housing_labels.iloc[:5]


# In[39]:


prepared_data= my_pipeline.transform(some_data)


# In[40]:


model.predict(prepared_data)


# In[41]:


list(some_labels)


# In[42]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
lin_mse= mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)


# In[43]:


lin_rmse


# In[44]:


from sklearn.model_selection import cross_val_score
scores= cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[45]:


rmse_scores


# In[46]:


def print_score(scores):
    print("Scores:", scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())
    


# In[47]:


print_score(rmse_scores)


# In[48]:


from joblib import dump, load
dump(model, "MLproject.joblib")


# In[50]:


X_test= strat_test_set.drop("MEDV",axis=1)
Y_test= strat_test_set["MEDV"].copy()
X_test_prepared= my_pipeline.transform(X_test)
final_predictions= model.predict(X_test_prepared)
final_mse= mean_squared_error(Y_test, final_predictions)
final_rmse= np.sqrt(final_mse)
final_rmse
print(list(final_predictions),list(Y_test))


# In[ ]:




