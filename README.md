

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
###
### <div align="center" style="color:rgb(200,200,90);font-size:36px"> &#8675; Project Notes: &#8675;</div>
- Python 3.10 or newer (specifically for [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html))
- Scikit-learn version 1.5.2 – to minimize FutureWarning errors when fitting. 
- When testing models independent of the Pipeline, ensure the data is first scaled!
- Links to source documentation:
  - For more on [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
  - For more on [LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
  - For more on [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  - For more on [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
  - For more on [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
  - For more on [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
  - For more on [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
  - For more on [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- Drink more coffee &#9749; &#9749; &#9749;

###
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
X = df.drop(columns=['Diagnosis','ID'])

# Verify expected shapes before and after:
print('X shape: ',X.shape)
print('y shape: ',y.shape)

# Convert target data to binary and verify value_counts.
print('\nTarget prior to binary conversion: \n',y.value_counts())
try:
    y = pd.DataFrame(np.where(y == 'M',1,0), columns=['Diagnosis'])
    y = y.Diagnosis
    print('\nTarget post binary conversion: \n',y.value_counts(),'\n')

except Exception as e:
    print(e)

# Verify expected shapes before and after:
print('X shape: ',X.shape)
print('y shape: ',y.shape)

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
```
###
### 4. Lasso L1 Regularization VS. LDA for Feature Selection / EDA &#9749;
```python
# Lasso outperforms LDA on feature selection as validated by SGDClassifier scores:
# Below are averaged results from fitting SGDClassifier 50 times using unique random_states
# and with the respective feature sets.

#                       Score       Std Dev     Features
# Lasso feature set:    0.9909      0.0017      19
# Full features set:	0.9889      0.0053      30
# LDA feature set:      0.9691      0.0054      19

# Both Lasso and LDA have 19 features after selection; however, Lasso features 
# outperform even the full data set (30 features), likely due to a reduction 
# in noisy data (i.e., noise from unimportant / low-importance features).
```
*For more on Lasso L1 Regularization see notebook blocks 4.1.X* <br>
*For more on LinearDiscriminantAnalysis see notebook blocks 4.2.X*

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
# to the best model's accuracy_score on testing data:
y_pred_gs = gs_best.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(gs.best_score_)

# Compare the RandomizedSearchCV best_score_ (training data) 
# to the best model's accuracy_score on testing data:
y_pred_rs = rs_best.predict(X_test)
print(accuracy_score(y_test, y_pred_rs))
print(rs.best_score_)
```
###
### 7. Hyperparameter Tuning and Model Retests
```python
# 7.1 Deep Dive into RandomForestClassifier Validation
# 7.2 Deep Dive into SGDClassifier Validation
# 7.3 RandomizedSearchCV Implementation / Runtime Benefits
# 7.4 Retesting with SMOTE - Synthetic Minority Oversampling Technique
# In Progress ...
```
###
### 7.1. Deep Dive into RandomForestClassifier Validation
*Takeaways:*
- RandomForestClassifier (RFC) is consistently underperforming (test scores: 92% full features, 97% Lasso features)
- RFC is significantly more expensive for run-time, averaging > 100 ms per fit compared to the 1 ms of SGDClassifier
###
### 7.2. Deep Dive into SGDClassifier Validation
*Takeaways:*
- SGDClassifier (SGDC) is a highly performant model (test scores: 98.9% full features, 99.1% Lasso features)
- SGDC performs well with the 'log_loss' and 'hinge' *loss* parameter, but 'log_loss' outperforms overall
- SGDC brings significant run-time benefits, averaging < 1 ms per fit and score.
- See block 7.4 for SGDC performance with SMOTE dataset balancing.
###
### 7.3. RandomizedSearchCV Implementation
*Takeaways:*
- RandomizedSearchCV has improved control of run-times due to the n_iter parameter which defaults to 10.
- Runs the risk of not finding the "best" parameters due to n_iter relative to search_space. 
- Ideal for quickly finding "good" parameters.
###
### 7.4. Retesting with SMOTE
*Takeaways:*
- SMOTE performs better on the Lasso feature set (19 features selected in blocks 4.1.X)
- SMOTE underperforms on the full feature set (30 features), likely due to an increase in sample noise
- BEST MODEL Accuracy is approaching 99.12% for the test set and 98% for the training set
```python
# Balances dataset with SMOTE by synthetic insertion of positive diagnosis samples
# into the training data set. Then scale and validate with SGDClassifier performance.
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=13,sampling_strategy=0.99,k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train,y_train)

# Note: Sampling_strategy only works when solving binary classification problems.
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
