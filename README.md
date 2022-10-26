# Loan-Classification-Exercise

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, ShuffleSplit, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, plot_confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
import category_encoders as ce

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
```

![download](https://user-images.githubusercontent.com/99155979/197922980-6741c6a7-f225-443e-b53a-47db80c76f1e.png)

**Analytic Approach**

This project goal is to create a predictive model to predict client late payment probability. The data will be analyze to obtain a pattern that differentiate between client who makes a late payment and client who doesn't make a late payment. The classification model will be formulated to prevent company to approved client who is going to make a late payment.

**Metric Evaluation**

Type 1 error : False Positive (Model predicts client makes a late payment while it is on time)/REJECTED

Consequence: losing potention revenue (admin commission fee 1% and 3% flat each month).
            
Type 2 error : False Negative (Model predicts client makes an on time payment while it is late)/APPROVED

Consequence: late revenue and additional cost for catching up the late revenue.

RECALL(+) CHURN

Based on the consequences, we are going to create a model that prevent company to lose potention revenue, but without adding more cost for catching up the late revenue by the client. So we have to balance later between precision and recall from the positive class (potential candidate). So later the main metric that we will use is roc_auc.

```python
df=pd.read_csv('ind_app_train.csv')

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 61503 entries, 0 to 61502
Data columns (total 24 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   Unnamed: 0         61503 non-null  int64  
 1   LN_ID              61503 non-null  int64  
 2   TARGET             61503 non-null  int64  
 3   CONTRACT_TYPE      61503 non-null  object 
 4   GENDER             61503 non-null  object 
 5   NUM_CHILDREN       61503 non-null  int64  
 6   INCOME             61503 non-null  float64
 7   APPROVED_CREDIT    61503 non-null  float64
 8   ANNUITY            61502 non-null  float64
 9   PRICE              61441 non-null  float64
 10  INCOME_TYPE        61503 non-null  object 
 11  EDUCATION          61503 non-null  object 
 12  FAMILY_STATUS      61503 non-null  object 
 13  HOUSING_TYPE       61503 non-null  object 
 14  DAYS_AGE           61503 non-null  int64  
 15  DAYS_WORK          61503 non-null  int64  
 16  DAYS_REGISTRATION  61503 non-null  float64
 17  DAYS_ID_CHANGE     61503 non-null  int64  
 18  WEEKDAYS_APPLY     61503 non-null  object 
 19  HOUR_APPLY         61503 non-null  int64  
 20  ORGANIZATION_TYPE  61503 non-null  object 
 21  EXT_SCORE_1        26658 non-null  float64
 22  EXT_SCORE_2        61369 non-null  float64
 23  EXT_SCORE_3        49264 non-null  float64
dtypes: float64(8), int64(8), object(8)
```

```python
desc = []
for i in df.columns:
    desc.append([
        i,
        df[i].dtypes,
        df[i].isna().sum(),
        round(((df[i].isna().sum() / len(df)) * 100), 2),
        df[i].nunique(),
        df[i].drop_duplicates().sample(2).values
    ])

pd.DataFrame(desc, columns=[
    "Data Features",
    "Data Types",
    "Null",
    "Null Percentage",
    "Unique",
    "Unique Sample"
])
```
<img width="528" alt="image" src="https://user-images.githubusercontent.com/99155979/197938083-87907787-5626-4316-a72b-aa521e6da308.png">


### Data Preprocessing
```python
df.drop(['Unnamed: 0','LN_ID','EXT_SCORE_1','EXT_SCORE_2','EXT_SCORE_3'], axis=1, inplace=True)
```
ID is being dropped because it has unique value for every row or data and also it doesn't have no effect on the target, external score also being dropped because it is a score from external data that we know nothing about.

```python
df = df[pd.notnull(df['ANNUITY'])]
df = df[pd.notnull(df['PRICE'])]
```
Drop missing values because the amount is small compared to the amount of data

```python
def minus(x):
    if x <0:
        return x * (-1)
    else:
        return x
```
```python
df['DAYS_AGE'] = df['DAYS_AGE'].apply(minus)
df['DAYS_WORK'] = df['DAYS_WORK'].apply(minus)
df['DAYS_REGISTRATION'] = df['DAYS_REGISTRATION'].apply(minus)
df['DAYS_ID_CHANGE'] = df['DAYS_ID_CHANGE'].apply(minus)
```
Convert minus value on the data

```python
plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot = True, cbar = False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197938993-d52ed64c-0a47-4a5b-9584-1b256161827f.png)

```python
import dython
from dython.nominal import associations, cramers_v, theils_u, correlation_ratio
```
```python
assoc_cr = []
col = ['NUM_CHILDREN', 'INCOME', 'APPROVED_CREDIT', 'ANNUITY', 'PRICE', 'DAYS_AGE', 'DAYS_WORK', 'DAYS_REGISTRATION', 'DAYS_ID_CHANGE', 'HOUR_APPLY']
for i in df.drop(columns = col).columns:
    assoc = round(cramers_v(df['TARGET'], df[i]), 2)
    assoc_cr.append(assoc)

df_cr = pd.DataFrame(data = [assoc_cr], columns = df.drop(columns = col).columns, index = ['TARGET'])

plt.figure(figsize = (15,1))
sns.heatmap(df_cr, annot = True, cbar = False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197939106-ec9143e9-84ac-414f-83f7-7ef6df08e2e9.png)

```python
pd.crosstab(df['TARGET'],columns='Percentage',normalize=True)*100
```
<img width="102" alt="image" src="https://user-images.githubusercontent.com/99155979/197939195-14147679-2b68-4a17-9392-db27bf833545.png">

Sekarang mari kita melakukan fitur encoding untuk fitur2 categorical yang kita miliki.
Yang akan kita lakukan adalah :

1. Merubah fitur/kolom `CONTRACT_TYPE` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
2. Merubah fitur/kolom `GENDER` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
3. Merubah fitur/kolom `INCOME_TYPE` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
4. Merubah fitur/kolom `EDUCATION` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan kita tidak tau berapa besar setiap jarak pastinya, maka akan lebih aman dan akurat bila kita menggunakan One Hot Encoding, selain itu jumlah unique datanya hanya sedikit.
5. Merubah fitur/kolom `FAMILY_STATUS` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
6. Merubah fitur/kolom `WEEKDAYS_APPLY` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
7. Merubah fitur/kolom `ORGANIZATION_TYPE` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.
8. Merubah fitur/kolom `HOUSING_TYPE` menggunakan One Hot Encoding, karena fitur ini tidak memiliki urutan/tidak ordinal, dan juga jumlah unique datanya hanya sedikit.

```python
transformer = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), ['CONTRACT_TYPE', 'GENDER', 'INCOME_TYPE', 'EDUCATION', 'FAMILY_STATUS', 'HOUSING_TYPE', 'WEEKDAYS_APPLY', 'ORGANIZATION_TYPE'])
], remainder='passthrough')

X = df.drop(columns=['TARGET'])
y = df['TARGET']

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=42)

logreg = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier()
```

```python
models = [logreg,knn,dt,rf,xgb]
score=[]
rata=[]
std=[]

for i in models:
    skfold=StratifiedKFold(n_splits=5)
    estimator=Pipeline([
        ('preprocess',transformer),
        ('model',i)])
    model_cv=cross_val_score(estimator,X_train,y_train,cv=skfold,scoring='roc_auc')
    score.append(model_cv)
    rata.append(model_cv.mean())
    std.append(model_cv.std())
    
pd.DataFrame({'model':['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'],'mean roc_auc':rata,'sdev':std}).set_index('model').sort_values(by='mean roc_auc',ascending=False)
```
<img width="207" alt="image" src="https://user-images.githubusercontent.com/99155979/197939459-c90c4e21-aa45-4baf-9f41-25c160e286db.png">
It can be seen that the XGBoost model is the best model for its roc_auc of any model that uses the default hyperparameter

```python
models = [logreg,knn,dt,rf,xgb]
score_roc_auc = []

def y_pred_func(i):
    estimator=Pipeline([
        ('preprocess',transformer),
        ('model',i)])
    X_train,X_test
    
    estimator.fit(X_train,y_train)
    return(estimator,estimator.predict(X_test),X_test)

for i,j in zip(models, ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost']):
    estimator,y_pred,X_test = y_pred_func(i)
    y_predict_proba = estimator.predict_proba(X_test)[:,1]
    score_roc_auc.append(roc_auc_score(y_test,y_predict_proba))
    print(j,'\n', classification_report(y_test,y_pred))
```

```python
pd.DataFrame({'model':['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'],
             'roc_auc score':score_roc_auc}).set_index('model').sort_values(by='roc_auc score',ascending=False)

Logistic Regression 
               precision    recall  f1-score   support

           0       0.92      1.00      0.96     11295
           1       0.00      0.00      0.00       993

    accuracy                           0.92     12288
   macro avg       0.46      0.50      0.48     12288
weighted avg       0.84      0.92      0.88     12288

KNN 
               precision    recall  f1-score   support

           0       0.92      0.99      0.95     11295
           1       0.17      0.02      0.03       993

    accuracy                           0.91     12288
   macro avg       0.55      0.51      0.49     12288
weighted avg       0.86      0.91      0.88     12288

Decision Tree 
               precision    recall  f1-score   support

           0       0.92      0.91      0.92     11295
           1       0.11      0.12      0.12       993

    accuracy                           0.85     12288
   macro avg       0.52      0.52      0.52     12288
weighted avg       0.86      0.85      0.85     12288

Random Forest 
               precision    recall  f1-score   support

           0       0.92      1.00      0.96     11295
           1       0.00      0.00      0.00       993

    accuracy                           0.92     12288
   macro avg       0.46      0.50      0.48     12288
weighted avg       0.84      0.92      0.88     12288

XGBoost 
               precision    recall  f1-score   support

           0       0.92      1.00      0.96     11295
           1       0.04      0.00      0.00       993

    accuracy                           0.92     12288
   macro avg       0.48      0.50      0.48     12288
weighted avg       0.85      0.92      0.88     12288
```
<img width="160" alt="image" src="https://user-images.githubusercontent.com/99155979/197940702-a9af6db7-5a12-4355-a0dd-4edfa7900c9e.png">
XGBoost model is still the best performing on the test data.
Next I will try to oversampling XGBoost model to see if we can get even better results.

### Test Oversampling with K-Fold Cross Validation
```python
def calc_train_error(X_train, y_train, model):
#     '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    predictProba = model.predict_proba(X_train)
    accuracy = accuracy_score(y_train, predictions)
    f1 = f1_score(y_train, predictions, average='macro')
    roc_auc = roc_auc_score(y_train, predictProba[:,1])
    recall = recall_score(y_train, predictions)
    precision = precision_score(y_train, predictions)
    report = classification_report(y_train, predictions)
    return { 
        'report': report, 
        'f1' : f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }
    
def calc_validation_error(X_test, y_test, model):
#     '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    predictProba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    roc_auc = roc_auc_score(y_test, predictProba[:,1])
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return { 
        'report': report, 
        'f1' : f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
#     '''fits model and returns the in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error
```
```python
K = 10
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
data = X_train
target = y_train
```
```python
train_errors_without_oversampling = []
validation_errors_without_oversampling = []

train_errors_with_oversampling = []
validation_errors_with_oversampling = []

for train_index, val_index in kf.split(data, target):
    
    # split data
    X_train_1, X_val = data.iloc[train_index], data.iloc[val_index]
    y_train_1, y_val = target.iloc[train_index], target.iloc[val_index]
    
#     print(len(X_val), (len(X_train) + len(X_val)))
    ros = RandomOverSampler()

    X_ros, y_ros = ros.fit_resample(X_train_1, y_train_1)

    # instantiate model
    estimator=Pipeline([
        ('preprocess',transformer),
        ('model',xgb)
    ])

    #calculate errors
    train_error_without_oversampling, val_error_without_oversampling = calc_metrics(X_train_1, y_train_1, X_val, y_val, estimator)
    train_error_with_oversampling, val_error_with_oversampling = calc_metrics(X_ros, y_ros, X_val, y_val, estimator)
    
    # append to appropriate list
    train_errors_without_oversampling.append(train_error_without_oversampling)
    validation_errors_without_oversampling.append(val_error_without_oversampling)
    
    train_errors_with_oversampling.append(train_error_with_oversampling)
    validation_errors_with_oversampling.append(val_error_with_oversampling)
```

#### Evaluation Metrics Without Oversampling
```python
listItem = []

for tr,val in zip(train_errors_without_oversampling,validation_errors_without_oversampling) :
    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],
                     tr['recall'],val['recall'],tr['precision'],val['precision']])

listItem.append(list(np.mean(listItem,axis=0)))
    
dfEvaluate = pd.DataFrame(listItem, 
                    columns=['Train Accuracy', 
                            'Test Accuracy', 
                            'Train ROC AUC', 
                            'Test ROC AUC', 
                            'Train F1 Score',
                            'Test F1 Score',
                            'Train Recall',
                            'Test Recall',
                            'Train Precision',
                            'Test Precision'])

listIndex = list(dfEvaluate.index)
listIndex[-1] = 'Average'
dfEvaluate.index = listIndex
dfEvaluate
```
<img width="563" alt="image" src="https://user-images.githubusercontent.com/99155979/197941089-00018b9f-aaa1-41e6-be34-c4f548464dce.png">


#### Evaluation Metrics With Oversampling
```python
listItem = []

for tr,val in zip(train_errors_with_oversampling,validation_errors_with_oversampling) :
    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],
                     tr['recall'],val['recall'],tr['precision'],val['precision']])

listItem.append(list(np.mean(listItem,axis=0)))
    
dfEvaluate = pd.DataFrame(listItem, 
                    columns=['Train Accuracy', 
                            'Test Accuracy', 
                            'Train ROC AUC', 
                            'Test ROC AUC', 
                            'Train F1 Score',
                            'Test F1 Score',
                            'Train Recall',
                            'Test Recall',
                            'Train Precision',
                            'Test Precision'])

listIndex = list(dfEvaluate.index)
listIndex[-1] = 'Average'
dfEvaluate.index = listIndex
dfEvaluate
```
<img width="560" alt="image" src="https://user-images.githubusercontent.com/99155979/197941143-fcccb774-9f8d-4670-9502-cb1454e008d4.png">
From the result above we got a better Recall(+) value after oversampling than before oversampling, otherwise for precision(+) value from the Evaluation Metrics result is got lower from before oversampling. It happend because there is a tradeoff on the increase of recall and the decrease of precision, also because the same amount of minority class data and majority class data.

```python
for rep in validation_errors_without_oversampling :
    print(rep['report'])
    
              precision    recall  f1-score   support

           0       0.92      1.00      0.96      4519
           1       0.33      0.01      0.01       397

    accuracy                           0.92      4916
   macro avg       0.63      0.50      0.48      4916
weighted avg       0.87      0.92      0.88      4916

              precision    recall  f1-score   support

           0       0.92      1.00      0.96      4519
           1       0.29      0.01      0.02       397

    accuracy                           0.92      4916
   macro avg       0.60      0.50      0.49      4916
weighted avg       0.87      0.92      0.88      4916

              precision    recall  f1-score   support

           0       0.92      1.00      0.96      4518
           1       0.22      0.01      0.01       397

    accuracy                           0.92      4915
   macro avg       0.57      0.50      0.48      4915
weighted avg       0.86      0.92      0.88      4915

              precision    recall  f1-score   support

           0       0.92      1.00      0.96      4518
           1       0.43      0.01      0.01       397

    accuracy                           0.92      4915
   macro avg       0.67      0.50      0.49      4915
weighted avg       0.88      0.92      0.88      4915

              precision    recall  f1-score   support

           0       0.92      1.00      0.96      4518
           1       0.50      0.01      0.01       397

    accuracy                           0.92      4915
   macro avg       0.71      0.50      0.49      4915
weighted avg       0.89      0.92      0.88      4915

              precision    recall  f1-score   support

           0       0.92      1.00      0.96      4518
           1       0.25      0.00      0.00       397

    accuracy                           0.92      4915
   macro avg       0.58      0.50      0.48      4915
weighted avg       0.87      0.92      0.88      4915

              precision    recall  f1-score   support

           0       0.92      1.00      0.96      4518
           1       0.00      0.00      0.00       397

    accuracy                           0.92      4915
   macro avg       0.46      0.50      0.48      4915
weighted avg       0.84      0.92      0.88      4915

              precision    recall  f1-score   support

           0       0.92      1.00      0.96      4518
           1       0.22      0.01      0.01       397

    accuracy                           0.92      4915
   macro avg       0.57      0.50      0.48      4915
weighted avg       0.86      0.92      0.88      4915

              precision    recall  f1-score   support

           0       0.92      1.00      0.96      4518
           1       0.18      0.01      0.01       397

    accuracy                           0.92      4915
   macro avg       0.55      0.50      0.48      4915
weighted avg       0.86      0.92      0.88      4915

              precision    recall  f1-score   support

           0       0.92      1.00      0.96      4518
           1       0.40      0.01      0.02       397

    accuracy                           0.92      4915
   macro avg       0.66      0.50      0.49      4915
weighted avg       0.88      0.92      0.88      4915
```

```python
for rep in validation_errors_with_oversampling :
    print(rep['report'])
    
              precision    recall  f1-score   support

           0       0.92      0.91      0.92      4519
           1       0.11      0.13      0.12       397

    accuracy                           0.85      4916
   macro avg       0.52      0.52      0.52      4916
weighted avg       0.86      0.85      0.85      4916

              precision    recall  f1-score   support

           0       0.92      0.92      0.92      4519
           1       0.12      0.13      0.12       397

    accuracy                           0.85      4916
   macro avg       0.52      0.52      0.52      4916
weighted avg       0.86      0.85      0.86      4916

              precision    recall  f1-score   support

           0       0.92      0.91      0.92      4518
           1       0.11      0.12      0.11       397

    accuracy                           0.85      4915
   macro avg       0.51      0.52      0.52      4915
weighted avg       0.86      0.85      0.85      4915

              precision    recall  f1-score   support

           0       0.92      0.91      0.91      4518
           1       0.09      0.11      0.10       397

    accuracy                           0.84      4915
   macro avg       0.51      0.51      0.51      4915
weighted avg       0.85      0.84      0.85      4915

              precision    recall  f1-score   support

           0       0.92      0.92      0.92      4518
           1       0.13      0.14      0.13       397

    accuracy                           0.85      4915
   macro avg       0.53      0.53      0.53      4915
weighted avg       0.86      0.85      0.86      4915

              precision    recall  f1-score   support

           0       0.92      0.91      0.92      4518
           1       0.13      0.15      0.14       397

    accuracy                           0.85      4915
   macro avg       0.53      0.53      0.53      4915
weighted avg       0.86      0.85      0.86      4915

              precision    recall  f1-score   support

           0       0.92      0.91      0.92      4518
           1       0.10      0.12      0.11       397

    accuracy                           0.84      4915
   macro avg       0.51      0.51      0.51      4915
weighted avg       0.86      0.84      0.85      4915

              precision    recall  f1-score   support

           0       0.92      0.91      0.92      4518
           1       0.10      0.12      0.11       397

    accuracy                           0.85      4915
   macro avg       0.51      0.51      0.51      4915
weighted avg       0.86      0.85      0.85      4915

              precision    recall  f1-score   support

           0       0.92      0.90      0.91      4518
           1       0.10      0.12      0.11       397

    accuracy                           0.84      4915
   macro avg       0.51      0.51      0.51      4915
weighted avg       0.86      0.84      0.85      4915

              precision    recall  f1-score   support

           0       0.92      0.91      0.92      4518
           1       0.10      0.12      0.11       397

    accuracy                           0.85      4915
   macro avg       0.51      0.51      0.51      4915
weighted avg       0.86      0.85      0.85      4915
```

### Hyperparameter Tuning
```python
pipe_XGB = Pipeline([
    ('prep', transformer),
    ('algo', XGBClassifier())
])

param_XGB = {
    "algo__n_estimators" : np.arange(50, 601, 50),
    "algo__max_depth" : np.arange(1, 10),
    "algo__learning_rate" : np.logspace(-3, 0, 4),
    "algo__gamma" : np.logspace(-3, 0, 6),
    "algo__colsample_bytree" : [0.3, 0.5, 0.7, 0.8],
    "algo__subsample" : [0.3, 0.5, 0.7, 0.8],
    "algo__reg_alpha" : np.logspace(-3, 3, 7),
    "algo__reg_lambda" : np.logspace(-3, 3, 7)
}

skf = StratifiedKFold(n_splits=10)
```
```python
GS_XGB = GridSearchCV(pipe_XGB, param_XGB, cv = skf, scoring='roc_auc', verbose = 3, n_jobs=-1)
RS_XGB = RandomizedSearchCV(pipe_XGB, param_XGB,cv = skf, scoring='roc_auc', verbose = 3, n_jobs=-1 )
RS_XGB.fit(X_train, y_train)
```

```python
RS_XGB.best_params_

{'algo__subsample': 0.7,
 'algo__reg_lambda': 0.1,
 'algo__reg_alpha': 100.0,
 'algo__n_estimators': 600,
 'algo__max_depth': 2,
 'algo__learning_rate': 1.0,
 'algo__gamma': 0.25118864315095796,
 'algo__colsample_bytree': 0.8}
 ```
 
 ```python
 XGB_Tuned = RS_XGB.best_estimator_
 
 print(classification_report(y_test, XGB_Tuned.predict(X_test)))
               precision    recall  f1-score   support

           0       0.92      1.00      0.96     11295
           1       0.00      0.00      0.00       993

    accuracy                           0.92     12288
   macro avg       0.46      0.50      0.48     12288
weighted avg       0.84      0.92      0.88     12288
```

```python
plot_confusion_matrix(XGB_Tuned, X_test, y_test, display_labels=['On time', 'Late'])
```
![download](https://user-images.githubusercontent.com/99155979/197944094-6450405e-f98f-4fee-8af7-e06a3007c418.png)

```python
coef1 = pd.Series(XGB_Tuned['algo'].feature_importances_, transformer.get_feature_names()).sort_values(ascending = False).head(10)
coef1.plot(kind='barh', title='Feature Importances')
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197944185-41a5c408-a011-43a9-bd98-301c67042274.png)

From the bar chart above from XGB model, the most important feature is `higher education`, follow by feature gender `M`, days work`, `Annuity`, and goes on. Other than that feature importance also can be done more thoroughly using Exploratory Data Analysis.
