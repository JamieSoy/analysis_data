#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostRegressor,CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings(action='ignore')


#  - enrollee_id : Unique ID for candidate
#  - city: City code
#  - city_development_index : Developement index of the city (scaled)
#  - gender: Gender of candidate
#  - relevent_experience: Relevant experience of candidate 
#  - enrolled_university: Type of University course enrolled if any
#  - education_level: Education level of candidate
#  - major_discipline :Education major discipline of candidate
#  - experience: Candidate total experience in years
#  - company_size: No of employees in current employer's company
#  - company_type : Type of current employer
#  - lastnewjob: Difference in years between previous job and current job
#  - training_hours: training hours completed
#  - target: 0 – Not looking for job change, 1 – Looking for a job change

# In[2]:


test = pd.read_csv('C:/KaggleData/HR-Analytics-Job-Change-of-Data-Scientists/aug_test.csv')
train = pd.read_csv('C:/KaggleData/HR-Analytics-Job-Change-of-Data-Scientists/aug_train.csv')


# In[3]:


print("="*20, "test","="*20)
display(test.info())
print("="*20, "train","="*20)
display(train.info())


# ## Train Dataset 

# In[4]:


##데이터 시각화
train.major_discipline = train.major_discipline.replace('Business Degree','Business')
fig, ((ax1, ax2), (ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(20,15))


gender = pd.crosstab(train['gender'],train['target'])
gender.plot(kind='bar',rot=0,ax=ax1,fontsize=15)
ax1.set_title("gender",fontsize=20)


Experience = pd.crosstab(train['relevent_experience'],train['target'])
Experience.plot(kind='bar',rot=0,ax=ax2,fontsize=15)
ax2.set_title("Experience",fontsize=20)


enrolled = pd.crosstab(train['enrolled_university'],train['target'])
enrolled.plot(kind='bar',rot=0,ax=ax3,fontsize=15)
ax3.set_title("enrolled_university",fontsize=20)

education = pd.crosstab(train['education_level'],train['target'])
education.plot(kind='bar',rot=0,ax=ax4,fontsize=15)
ax4.set_title("Education",fontsize=20)

major = pd.crosstab(train['major_discipline'],train['target'])
major.plot(kind='bar',rot=0,ax=ax5,fontsize=15)
ax5.set_title("Major",fontsize=20)


Company = pd.crosstab(train['company_type'],train['target'])
Company.plot(kind='bar',rot=0,ax=ax6,fontsize=15)
ax6.set_title("Company_type",fontsize=20)


plt.tight_layout()
plt.show()


# In[5]:


# Target data insight
plt.figure(figsize=(10, 5))
explode=[0.0,0.1]
labels=['Not looking for job change','looking for job change']
plt.pie(train['target'].value_counts(),explode=explode,startangle=45,labels=labels,autopct='%.1f%%')
plt.title("Target",fontsize=20)
plt.tight_layout()
plt.show()


# In[6]:


print("="*20, "train","="*20)
for col in train.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (train[col].isnull().sum() / train[col].shape[0]))
    print(msg)
    
print("="*20, "test","="*20)
for col in test.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (test[col].isnull().sum() / test[col].shape[0]))
    print(msg)


# In[7]:


# null 값 확인

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20,5))

msno.matrix(df=train.iloc[:, :],ax=ax[0])
msno.matrix(df=test.iloc[:, :],ax=ax[1])
ax[0].set_title("train",fontsize=30)
ax[1].set_title("test",fontsize=30)

plt.show()


# ### Handling Features (Labeling)

# In[16]:


# NaN 처리먼저
# 그 후 라벨링
# 상관분석 및 변수시각화


# In[3]:


# https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                # convert float NaN --> string NaN
                output[col] = output[col].fillna('NaN')
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[4]:


# store the catagorical features names as a list      
cat_features = train.select_dtypes(['object']).columns.to_list()
# use MultiColumnLabelEncoder to apply LabelEncoding on cat_features 
# uses NaN as a value , no imputation will be used for missing data
train_transform = MultiColumnLabelEncoder(columns = cat_features).fit_transform(train)
test_transform = MultiColumnLabelEncoder(columns = cat_features).fit_transform(test)


# In[97]:


display(train)
display(train_transform)


# In[99]:


display(test)
display(test_transform)


# ## Train-test split ( train dataset)
# 

# In[5]:


x = train_transform.iloc[:,1:13]
y = train_transform.iloc[:,13]
train_x, val_x , train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=123, stratify=y )


# # Modeling

# ### Modeling1 - Random Forest

# In[22]:


rfc = RandomForestClassifier()
model_rfc = rfc.fit(train_x,train_y)


# ### Modeling2 - LightGBMClassifier using GridSearch

# In[9]:


lgbm= LGBMClassifier()
model = lgbm.fit(train_x, train_y)
param_grid = {
 'learning_rate': [0.01, 0.1, 0.05, 0.5, 1],
    'n_estimators': [20, 40, 60, 80, 100, 120]
}
grid = GridSearchCV(estimator=lgbm, scoring='roc_auc',param_grid=param_grid, cv=5)

grid.fit(train_x, train_y)
print('best score=',grid.best_score_)
print('best parameter=',grid.best_params_)


# In[33]:


lgbm= LGBMClassifier(learning_rate=0.1, n_estimators= 40)
model_lgbm = lgbm.fit(train_x, train_y)


# ### Modeling3 - CatBoostClassifier using GridSearch

# In[12]:


model_CBC = CatBoostClassifier()
 
parameters = {'depth'         : [6,8,10],
              'learning_rate' : [0.01, 0.05, 0.1],
             'iterations'    : [30, 50, 100]
                 }
grid = GridSearchCV(estimator=model_CBC, param_grid = parameters, cv = 2, n_jobs=-1)
grid.fit(train_x, train_y)
best_param = grid.best_estimator_

print("best score : ", grid.best_score_)
print("best parameters : ", grid.best_params_)


# In[13]:


cat = CatBoostClassifier(depth= 8, iterations= 100, learning_rate= 0.1)
model_cat = cat.fit(train_x, train_y)


# ### Models evaluation

# In[29]:


rfc_y_pred = model_rfc.predict(val_x)
acc_rfc = accuracy_score(val_y,rfc_y_pred)

lgbm_y_pred = model_lgbm.predict(val_x)
acc_lgbm = accuracy_score(val_y,lgbm_y_pred)


catc_y_pred = model_cat.predict(val_x)
acc_catc = accuracy_score(val_y,catc_y_pred)

print("="*15, "Models Accuravy","="*15)
print("Random Forest : ",acc_rfc)
print("LightGBM : ",acc_lgbm)
print("CatBoostClassifier : ",acc_catc)


# In[31]:


fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(val_y, rfc_y_pred)
auc_rfc = auc(fpr_rfc, tpr_rfc)

fpr_lgbm, tpr_lgbm, thresholds_lgbm = roc_curve(val_y, lgbm_y_pred)
auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

fpr_catc, tpr_catc, thresholds_catc = roc_curve(val_y, catc_y_pred)
auc_catc = auc(fpr_catc, tpr_catc)

print("="*15, "Models AUC","="*15)
print("Random Forest : ",auc_rfc)
print("LightGBM : ",auc_lgbm)
print("CatBoostClassifier : ",auc_catc)


# In[32]:


fig = plt.figure(figsize = (12, 8))
chart = fig.add_subplot(1,1,1)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_rfc, tpr_rfc,'r', label = 'RondomForest')
plt.plot(fpr_lgbm, tpr_lgbm,'y', label = 'LightGBM')
plt.plot(fpr_catc, tpr_catc,'g', label = 'CatBoostClassifier')


chart.set_title('Models ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='best')


# # Suggestion

# In[46]:


su_test = test_transform.drop(['enrollee_id'],axis=1)


# In[50]:


y_predict = model_lgbm.predict(su_test)
suggestion = pd.DataFrame({'enrollee_id' : test_transform['enrollee_id'],'target':y_predict})
suggestion = suggestion.sort_values(by='target' ,ascending=False)


# In[53]:


suggestion


# # Review

# - 개인적으로 결론만 보면 분류모델보다는 확률로 나타낼 수 있는 회귀모델이 적합하다고 생각한다.
# 이 데이터의 경우, 종속변수가 Binary의 형태이기때문에 분류모델로 모델링을 해주어야 정확도가 그나마 80%가까이 나올 수 있다.
# 중간에 시간을 투자하여 catboost 를 이용하여 regression 모델을 적용해보았으나 정확도가 24%정도여서 모델을 사용할 수 없었다.
# regression 모델을 사용하고 싶었던 이유는, 확률에 따라서 결과를 표현하는게 적합하다고 생각해서였다.
# binary인경우 회귀모델은 쓸 수는 없는지, 또한 사용하기 위해서 어떤 처리과정을 추가해주어야하는지 추후에 시간이 있으면 다시 시도해보려고 한다.
# 
# - 모델링을 세개를 적용해주면서, 생각보다 모델 정확도가 낮아서 그 이유는 무엇인지 찾아보았다. 
# 아마 overfitting이 문제였던 것 같아서. k_fold cross validation을 이용하였다. 하지만 그렇다고 하여 정확성이 크게 차이가 나지는 않았다. 다른 캐글러의 글들을 보며 조금씩 더 공부를 해야할 것 같다. 
# 
# - Null값을 다루기 위해 3개의 imputor를 사용하였다. 그 중  MI적용시킨 labelencoding 을 이용하였다.(stackflow 참조) 본 노트북에는 없지만, 다른 두개의 imputor는 knn과 bayesian 을 이용하였으나,  Knn은 데이터적중률이 낮아지는 듯하고 baysian은 시간관계상 공부할 시간이 없어서 적용하다가 미뤄뒀다. 
# 
# - 모델들의 검증방법을 자세히 알고 적용할 수 있었던 좋은 데이터였다.
# 
# 

# ### self  assignment

# - Bayesian Imputor 
# - Why decrease accuracy?
# - How come increse model's accuracy?
