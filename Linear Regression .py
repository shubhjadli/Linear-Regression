#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data=pd.read_csv(r'/Users/shubhamjadli/Downloads/SeoulBikeData.csv')
data.columns=['Date', 'Rented Bike Count', 'Hour', 'Temperature', 'Humidity(%)',
       'Wind speed_(m/s)', 'Visibility_(10m)', 'Dew_point_temperature(C)',
       'Solar_Radiation_(MJ/m2)', 'Rainfall(mm)', 'Snowfall(cm)', 'Seasons',
       'Holiday', 'Functioning_Day']


# In[21]:


print('NA values=\n',data.isna().sum())
print('Duplicated values=\n',data.duplicated().sum())


# In[23]:


data.corr()['Temperature'].sort_values(ascending=False)


# In[24]:


x=data.Holiday[23]
data.Seasons=data.Seasons.map({'Summer':0, 'Spring':1,'Autumn':2,'Winter':3})
data.Holiday=data.Holiday.map({'Holiday':1,x:0})
data.Functioning_Day=data.Functioning_Day.map({'Yes':0,'No':1})


# In[25]:


plt.figure(figsize=(25,25))
sns.heatmap(data.corr(), center=0, annot=True)
plt.title("Correlation Map")
plt.show()


# In[26]:


num_atr=['Rented Bike Count', 'Hour', 'Temperature', 'Humidity(%)','Wind speed_(m/s)', 'Visibility_(10m)', 'Dew_point_temperature(C)','Solar_Radiation_(MJ/m2)', 'Rainfall(mm)', 'Snowfall(cm)']
cat_atr=['Seasons','Holiday']


# In[27]:


data[num_atr].hist(bins=50, figsize=(20,15)) 
plt.show()


# In[28]:


data.head()


# In[29]:


ss = StandardScaler()
X= data.drop(['Date','Functioning_Day'],axis=1)
Y= data['Functioning_Day']
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=40)
ss.fit_transform(X_train[num_atr])
X_train[num_atr] = ss.fit_transform(X_train[num_atr])
X_train[num_atr].head()


# In[30]:


X_train


# In[31]:


le = LabelEncoder()
X_train[cat_atr] = X_train[cat_atr].apply(le.fit_transform)
X_train[cat_atr].head()


# In[32]:


z=X_train[cat_atr]
y=X_train[num_atr]
print (z.head())
print (y.head())
CCT= pd.concat([z,y], axis=1)
CCT.head()


# In[33]:


reg = RandomForestRegressor()
reg.fit(CCT, y_train)
X_test[num_atr] = ss.fit_transform(X_test[num_atr])
le = LabelEncoder()
X_test[cat_atr] = X_test[cat_atr].apply(le.fit_transform)


# In[38]:


c=X_test[cat_atr]
d=X_test[num_atr]
CCT2=pd.concat([c,d], axis=1)
reg.predict(CCT2)
from sklearn.model_selection import cross_val_score
def check(data,X_train,y_train):
    rmse_score=cross_val_score(data,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
    r2_score=cross_val_score(data,X_train,y_train,scoring='r2',cv=10)
    print("RMSE (cross_val_score): ",np.sqrt(-rmse_score).mean())
    print("R2 Score (cross_val_score): ",r2_score.mean())


# In[39]:


print('Test variables shape=\n',X_test.shape)
print('Test variables o/p shape=\n', y_test.shape)
print('Concatted categorical and numerical variable shape=\n',CCT2.shape)


# In[40]:


check(reg,CCT2,y_test)

