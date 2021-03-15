#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection  import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import collections


# In[2]:


credit_raw = pd.read_csv("C:/KaggleData/BankChurners.csv")
credit = credit_raw[credit_raw.columns[:-2]]

credit.info()


# In[3]:


credit.describe()


# In[4]:


labels = ['Exist','Attirited']
plt.pie(credit['Attrition_Flag'].value_counts(),labels=labels,autopct='%.1f%%')
plt.title("Attrition_Flag",fontsize=20)
plt.show()


# In[5]:


fig, ((ax1, ax2), (ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

df1 = pd.crosstab(credit['Attrition_Flag'],credit['Dependent_count'])
df1.plot(kind = 'bar', title = 'Dependent_count',ax=ax1,rot=0)

df2 = pd.crosstab(credit['Attrition_Flag'],credit['Education_Level'])
df2.plot(kind = 'bar',title = 'Education_Level',ax=ax2,rot=0)

df3 = pd.crosstab(credit['Attrition_Flag'],credit['Income_Category'])
df3.plot(kind = 'bar',title = 'Income_Category',ax=ax3,rot=0)

df4 = pd.crosstab(credit['Attrition_Flag'],credit['Card_Category'])
df4.plot(kind = 'bar',title = 'Card_Category',ax=ax4,rot=0)

df5 = pd.crosstab(credit['Attrition_Flag'],credit['Gender'])
df5.plot(kind = 'bar',title = 'Card_Gender',ax=ax5,rot=0)

df6 = pd.crosstab(credit['Attrition_Flag'],credit['Marital_Status'])
df6.plot(kind = 'bar',title = 'Marital_Status',ax=ax6,rot=0)

plt.show()


# In[6]:


plt.rcParams['figure.figsize'] = [5,5]
sns.boxplot( data= credit,x='Attrition_Flag', y='Customer_Age')

plt.tight_layout()
plt.show()


# 인구학적 데이터으로는 고객이탈과 상관이 없음. 데이터의 모양이 같으므로 확인 가능. 그렇다면, 어떤 전제조건들로 고객이탈을 예측할 수 있을지 다른 변수들로 확인보겠음.

# In[9]:


dummy_att = pd.get_dummies(credit['Attrition_Flag'])
df = credit.iloc[:,9:21]
con_df = df.join(dummy_att)
con_df.head()


# NonNuneric data와 종속변수의 상관성을 confusion Matrix로 나타냄

# In[10]:


corr=con_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(240, 20, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# In[11]:


df = credit.iloc[:,9:21]
attrition = credit['Attrition_Flag']
con_df = df.join(attrition)
con_df.head()


# # Modeling(XGBoost, SVM, Random Forest with Random SearchCV)

# In[12]:


X = con_df.iloc[:,0:12]
y = con_df.iloc[:,12]


def normal(x) :
    nor = (x - np.min(x)) / (np.max(x) - np.min(x))
    return nor

X_nor = normal(X)

X_train, X_test, y_train, y_test = train_test_split(X_nor, y, test_size=0.3, random_state=123)


# In[48]:


# RandomForest

rfc=RandomForestClassifier(n_estimators=100)
model = rfc.fit(X=X,y=y)

params = {'n_estimators':[100,150,200,300],
          'max_depth':[None, 3, 5, 7],
          'min_samples_leaf':[3,5,7,9],
          'min_samples_split':[2,3,4,5],
          'max_features':['auto','sqrt']} # dict

randomized = RandomizedSearchCV(estimator=model, param_distributions = params ,
             scoring='accuracy', cv=5, n_jobs=-1)

randomized_model = randomized.fit(X,y)

print('best score =',randomized_model.best_score_) 
print('best parameter =',randomized_model.best_params_)
print('#'*50)

rfc=RandomForestClassifier(n_estimators = 100, 
                           min_samples_split= 4, min_samples_leaf= 3, max_features= 'sqrt', max_depth= None)
rfc_model = rfc.fit(X=X_train, y=y_train)
y_rfc = rfc_model.predict(X = X_test)
y_true = y_test

report = classification_report(y_test, y_rfc)
print('Random Forest with Random Search Parameters Report')
print(report)
print('-'*50)


# XGBoost

obj = XGBClassifier() #default params
xgb_model = obj.fit(X=X_train, y=y_train)
y_xgb = xgb_model.predict(X_test)
y_true= y_test

report = classification_report(y_test, y_xgb)
print('XGBoost Report')
print(report)
print('-'*50)


# RBF SVM
obj = SVC() 
svm_model = obj.fit(X=X_train, y=y_train)
y_svm = svm_model.predict(X = X_test)
y_true = y_test

report = classification_report(y_test, y_svm)
print('SVM Report')
print(report)
print('-'*50)


# In[36]:


fig,ax=plt.subplots(ncols=3, figsize=(20,6))

con_mat_rfc = confusion_matrix(y_true, y_rfc)
labels = ["Attrited","Existing"]
sns.heatmap(con_mat_rfc,annot=True,xticklabels=labels, 
            yticklabels=labels, fmt='d', annot_kws={"size": 15},linewidths=.5,ax=ax[0])
ax[0].title.set_text('RF-Random Search Confusion Matrix')

con_mat_xgb = confusion_matrix(y_true, y_xgb)
labels = ["Attrited","Existing"]
sns.heatmap(con_mat_xgb,annot=True,xticklabels=labels, 
            yticklabels=labels, fmt='d', annot_kws={"size": 15},linewidths=.5,ax=ax[1])
ax[1].title.set_text('XGBoost Confusion Matrix')

con_mat_svm = confusion_matrix(y_true, y_svm)
labels = ["Attrited","Existing"]
sns.heatmap(con_mat_svm,annot=True,xticklabels=labels, 
            yticklabels=labels, fmt='d', annot_kws={"size": 15},linewidths=.5,ax=ax[2])
ax[2].title.set_text('SVM Confusion Matrix')

plt.show()


# ML Feature Importance Visualizaion

# In[55]:


rf_features_f1 = randomized_model.best_estimator_.feature_importances_
rf_features = pd.DataFrame({'importance':rf_features_f1}, index=X.columns)
rf_features.sort_values(by = 'importance', ascending = True , axis =0).plot(kind='barh',figsize=(10,5))

plt.xlabel("Feature Importance by RF")
plt.title("RF Feature Importance")
plt.show()


fscore = xgb_model.get_booster().get_fscore()
plot_importance(booster=xgb_model)
plt.title("XGBoost Feature Importance")
plt.show()


# Based on XGBoost result (higher than RF) , We can choose Top5 Features.
# 
# Total_Trans_Amt : Total Transaction Amount(Last 12 months)
# Total_Amt_Chung_Q4_Q1 : Change of relationship with bank(Q4 over Q1)
# Total_Trans_Ct : Total Transacion Count(Last 12 months)
# Total_Ct_Chng_Q4_Q1 : Change in Transaction Count(Q4 over Q1)
# Months_on_book : Period of relationship with bank
# 

# # Conclusion

# - Bank needs to some Transaction Marketing for potantial attrition customer.
# - As the number of Transaction decreases, The more likely it is that customers will churn. 
# 

# In[ ]:




