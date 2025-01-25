#!/usr/bin/env python
# coding: utf-8

# Importing the librarires 
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import  SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import warnings


# Loading the dataset

# In[2]:


#reading the dataset
df=pd.read_csv(r'C:\Users\itzsa\OneDrive\Desktop\Project\insurance.csv')


# Data overview

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


#looking at the statstical information 
df.describe()


# In[8]:


df.info()


# Data analysis

# In[9]:


sns.set()
plt.figure(figsize=(7,7))
sns.histplot(data=df, x='age', kde=True)
plt.title('Age')


# In[10]:


plt.figure(figsize=(7,7))
sns.countplot(data=df, x='sex')
plt.title('Gender')


# In[11]:


df['sex'].value_counts()


# In[12]:


sns.set()
plt.figure(figsize=(7,7))
sns.histplot(data=df, x='bmi', kde=True)
plt.title('bmi')


# normal bmi range -- 18.5-25,
# Underweight-- <18.5,
# Overweight-- >25

# In[13]:


plt.figure(figsize=(7,7))
sns.countplot(data=df, x='children')
plt.title('Birth')


# In[14]:


df['children'].value_counts()


# In[15]:


plt.figure(figsize=(7,7))
sns.countplot(data=df, x='smoker')
plt.title('smoker')


# In[16]:


df['smoker'].value_counts()


# In[17]:


plt.figure(figsize=(7,7))
sns.histplot(data=df, x='charges', kde=True)
plt.title('bmi')


# In[18]:


df


# Label Encoding categorical values

# In[19]:


le = LabelEncoder()


# In[20]:


df['sex'] = le.fit_transform(df['sex'])
#df['sex'].replace({'sex':{'male':1,'female':0}},inplace=True)
#df['sex'].replace({'male': 1, 'female': 0}, inplace=True)
#df['sex'] = df['sex'].replace({'male': 1, 'female': 0})



# In[21]:


df['smoker'] = le.fit_transform(df['smoker'])
#df['smoker'].replace({'yes':1,'no':0},inplace=True)
#df['smoker'] = df['smoker'].replace({'yes': 1, 'no': 0})


# In[22]:


df['region'] = le.fit_transform(df['region'])
#df['region'].replace({'southeast':1,'southwest':2,'northeast':3,'northwest':4},inplace=True)
#df.replace({'region': {'southeast': 1, 'southwest': 2, 'northeast': 3, 'northwest': 4}}, inplace=True)



# In[23]:


df


# In[24]:


df.info()


# Defining parameters

# In[25]:


X = df.drop(columns='charges',axis = 1,)
Y = df['charges']


# In[26]:


X


# In[27]:


Y


# Initializing Train-Test-Split

# In[28]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=40)


# In[29]:


print(X.shape,X_test.shape,X_train.shape)


# In[30]:


print(Y.shape,Y_test.shape,Y_train.shape)


# Model Training 

# In[31]:


m1=LinearRegression()


# In[32]:


m1.fit(X_train,Y_train)


# In[33]:


m2=SVR()


# In[34]:


m2.fit(X_train,Y_train)


# In[35]:


m3=RandomForestRegressor()


# In[36]:


m3.fit(X_train,Y_train)


# In[37]:


m4=GradientBoostingRegressor()


# In[38]:


m4.fit(X_train,Y_train)


# In[39]:


train_predict1=m1.predict(X_train)
train_predict2=m2.predict(X_train)
train_predict3=m3.predict(X_train)
train_predict4=m4.predict(X_train)


# In[40]:


test_predict1=m1.predict(X_test)
test_predict2=m2.predict(X_test)
test_predict3=m3.predict(X_test)
test_predict4=m4.predict(X_test)


# Performance metrics

# In[41]:


r_2train1=metrics.r2_score(Y_train,train_predict1)
print('r squared value for training data : ',r_2train1)
r_2train2=metrics.r2_score(Y_train,train_predict2)
print('r squared value for training data : ',r_2train2)
r_2train3=metrics.r2_score(Y_train,train_predict3)
print('r squared value for training data : ',r_2train3)
r_2train4=metrics.r2_score(Y_train,train_predict4)
print('r squared value for training data : ',r_2train4)


# In[42]:


r_2test1=metrics.r2_score(Y_test,test_predict1)
print('r squared value for testing data : ',r_2test1)
r_2test2=metrics.r2_score(Y_test,test_predict2)
print('r squared value for testing data : ',r_2test2)
r_2test3=metrics.r2_score(Y_test,test_predict3)
print('r squared value for testing data : ',r_2test3)
r_2test4=metrics.r2_score(Y_test,test_predict4)
print('r squared value for testing data : ',r_2test4)


# In[43]:


mae1 =metrics.mean_absolute_error(Y_test,test_predict1)
print('r squared value for testing data : ',mae1)
mae2 =metrics.mean_absolute_error(Y_test,test_predict2)
print('r squared value for testing data : ',mae2)
mae3 =metrics.mean_absolute_error(Y_test,test_predict3)
print('r squared value for testing data : ',mae3)
mae4 =metrics.mean_absolute_error(Y_test,test_predict4)
print('r squared value for testing data : ',mae4)


# In[44]:


df1 = pd.DataFrame({'Actual':Y_test,'m1':test_predict1,'m2':test_predict2,'m3':test_predict3,'m4':test_predict4})


# In[45]:


df1


# In[46]:


plt.subplot(221)
plt.plot(df1['Actual'].iloc[0:10],label='Actual')
plt.plot(df1['m1'].iloc[0:10],label='m1')

plt.subplot(222)
plt.plot(df1['Actual'].iloc[0:10],label='Actual')
plt.plot(df1['m2'].iloc[0:10],label='m2')

plt.subplot(223)
plt.plot(df1['Actual'].iloc[0:10],label='Actual')
plt.plot(df1['m3'].iloc[0:10],label='m3')

plt.subplot(224)
plt.plot(df1['Actual'].iloc[0:10],label='Actual')
plt.plot(df1['m4'].iloc[0:10],label='m4')


# In[ ]:





# In[ ]:





# FINAL PREDICTION FOR LINEAR REGRESSION
# 

# In[48]:


#Prediction_num = (AGE,GENDER,BMI,CHILDREN,SMOKER,REGION)
data = {'age':40,
        'sex':1,
        'bmi':40.30,
        'children':4,
        'smoker':1,
        'region':2}
DF = pd.DataFrame(data,index=[0])


# In[49]:


new_pred = m4.predict(DF)
print(new_pred)


# Traning on entire dataset

# In[50]:


ent_data = GradientBoostingRegressor()
ent_data.fit(X,Y)


# In[51]:


import joblib as jb


# In[52]:


jb.dump(ent_data,'ent_data.joblib')


# In[53]:


model = jb.load('ent_data.joblib')


# In[54]:


model.predict(DF)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[265]:


#Prediction_ary =np.asarray(Prediction_num)
#Prediction_ary


# In[266]:


#Prediction = Prediction_ary.reshape(1,-1)
#Prediction


# In[267]:


FINALPREDICTION =m1.predict(Prediction)
print(FINALPREDICTION)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




