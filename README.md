

# <div align="center" style="color:rgb(200,180,180)"> ***UCI Wisconsin Breast Cancer (Diagnostic) Dataset*** </div>
###
## <div align="center" style="color:rgb(130,170,170);font-size:48px"> &#9733;  ML Pipelines Project  &#9733;

###
### <div align="center" style="color:rgb(200,200,90);font-size:36px"> &#8675; Project Introduction: &#8675;</div>

- The Breast Cancer (Diagnostic) Dataset can be found on the [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Project focus is the implementation of ML Pipelines for binary classification, to include: 
  - Data preprocessing and transformations (i.e., scale, impute, convert)
  - Feature selection and validation (i.e., Lasso Regularization or LDA)
  - Optimization of model parameters (search_space)
  - Comparing resultant models of GridSearchCV and RandomizedSearchCV
  - Developing a scalable and maintainable ML Pipeline for binary classification
  - Validating the model with "unseen" testing data

###
### <div align="center" style="color:rgb(200,200,90);font-size:36px"> &#8675; Project Notes: &#8675;</div>
- Scikit-learn version 1.5.2 – to minimize FutureWarning errors when fitting. 
- When testing models independent of the Pipeline, ensure the data is first scaled!
- Drink more coffee &#9749; &#9749; &#9749;

###
### <div align="center" style="color:rgb(200,200,90);font-size:36px"> &#8675; Project Overview: &#8675;</div>


### 1. Import dataset and write to local csv:  &#9921; &#9985; &#9998;
```python
## Import UCI Dataset and write to local csv
from ucimlrepo import fetch_ucirepo
breast_ca = fetch_ucirepo(id=17)

breast_ca_df = breast_ca.data.original
breast_ca_df.to_csv('UCI_BreastCancer.csv', index=False)
print('Successfully wrote dataset to csv file!')
```
*<div style="margin-left: 60px;"> OR: </div>*
```python
## Import UCI Dataset from sklearn.datasets and write to local csv
from sklearn.datasets import load_breast_cancer
breast_ca = load_breast_cancer()

breast_ca.to_csv('UCI_BreastCancer.csv', index=False)
print('Successfully wrote dataset to csv file!')
```
###
### 2. Review data for completeness and feature data types 
```python
# Search Dataset for missing / null values
try:
    if df.isnull().sum().any()>0:
        print('NaN values found: ', df.isnull().sum())
    else:
        print('No NaN or null values found')
except Exception as e:
    print(e)
    
# Consider the number of unique values for each feature:
print(df.nunique())

# Verify features and shape
print(df.columns)
print(df.shape)
```
###
### 3. Define Target and Features, Convert Target to Binary (0,1), Split data
```python
# Define features and target
y = df.Diagnosis
X = df.drop(columns=['Diagnosis'])

# Convert target data to binary and verify value_counts.
print('\nPrior to binary conversion: \n',y.value_counts())
try:
    y = pd.DataFrame(np.where(y == 'M',1,0), columns=['Diagnosis'])
    y = y.Diagnosis
    print('\nPost binary conversion: \n',y.value_counts(),'\n')

except Exception as e:
    print(e)

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
```
###
### 4. Lasso L1 Regularization VS. LDA for Feature Selection / EDA &#9749;
```python
# Lasso outperforms LDA on feature selection as validated by SGDClassifier scores:
# Below are averaged results from fitting SGDClassifier 50 times 
# with the cooresponding feature sets.

#                       Score       Std Dev
# Lasso feature set:    0.9846      0.0065
# LDA feature set:      0.9654      0.0094
```
###
### 5. Define Pipeline and search_space for model selection in the next goal
```python
# Define Preprocessor, Pipeline, and search_space for GridSearchCV()
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ('scaler', scaler, X_train.columns)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', ClfSwitch())
])

search_space = [
    {'clf__estimator': [RandomForestClassifier(random_state=13)],
     'clf__estimator__max_depth':[10,15,25],
     'clf__estimator__n_estimators':[150,200,250],
     },
    {'clf__estimator': [GradientBoostingClassifier(random_state=13)],
     'clf__estimator__learning_rate':[0.001,0.01,0.1,0.5],
     'clf__estimator__n_estimators':[150,200,250],
     },
    {'clf__estimator': [SGDClassifier(random_state=13)],
     'clf__estimator__loss': ['hinge','log_loss'],
     'clf__estimator__alpha': [0.01],
     'clf__estimator__penalty': ['l2']
     }
]
```
###
### 6. Use GridSearchCV or RandomizedSearchCV to fit the model 
```python
from sklearn.model_selection import GridSearchCV

# Model Selection and Hyperparameter tuning:
gs = GridSearchCV(estimator=pipeline, param_grid=search_space, cv=5, error_score='raise')
gs.fit(X_train, y_train)

# Load the best performing model as gs_best:
gs_best = gs.best_estimator_

# Load the best performing model classifier as gs_best_clf:
gs_best_clf = gs_best.named_steps['clf']

# Print the estimator that best fit the data with the given search_space:
print(gs_best_clf.get_params()['estimator'])
```
*<div style="margin-left: 60px;"> OR: </div>*
```python
from sklearn.model_selection import RandomizedSearchCV

# Model Selection and Hyperparameter tuning:
rs = RandomizedSearchCV(estimator=pipeline, param_distributions=search_space, 
                        cv=5, n_iter=5, error_score='raise')
rs.fit(X_train, y_train)

# Load the best performing model as rs_best:
rs_best = rs.best_estimator_

# Load the best performing model classifier as rs_best_clf:
rs_best_clf = rs_best.named_steps['clf']

# Print the estimator that best fit the data with the given search_space:
print(rs_best_clf.get_params()['estimator'])
```
###
### 6.1. Feature analysis and hyperparameter tuning
```python
# Using gs.cv_results_ for model parameter tuning:
cv_df = pd.DataFrame(gs.cv_results_)

columns_of_interest = [
    'param_clf__estimator',
    'param_clf__estimator__max_depth',
    'param_clf__estimator__n_estimators',
    'param_clf__estimator__loss',
    'param_clf__estimator__penalty',
    'param_clf__estimator__alpha',
    'mean_test_score',
    'std_test_score',
    'rank_test_score']

cv_df_results = cv_df[columns_of_interest].round(3)
cv_df_results.style.background_gradient(axis=0,cmap='Spectra')

print(cv_df_results)
```
###
### 6.2. Validate model performance
```python
from sklearn.metrics import accuracy_score

# Compare the GridSearchCV best_score_ (training data)
# to the best model's accuracy_score (testing data):
y_pred_gs = gs_best.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(gs.best_score_)

# Compare the RandomizedSearchCV best_score_ (training data) 
# to the best model's accuracy_score (testing data):
y_pred_rs = rs_best.predict(X_test)
print(accuracy_score(y_test, y_pred_rs))
print(rs.best_score_)
```
###
### 7. Hyperparameter Tuning and Model Retests
```python
# In Progress ...
```
###
### 8. Scale and implement Pipeline as a flexible/alterable model that can adapt  &#9749;
```python
# In Progress: ...
```
#
#

<!--
Please don't delete
&#9733; &#9968; &#9749; &#10004; &#10008; &#9921; &#9985; &#9998;
-->
