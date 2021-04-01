"""
Do model selection by 10-fold CV:

- Load data, combine features from preprocessing
- Split data: test_size=0.8, because time is of the essence
- Test basic model
- Do random search CV for log. reg., SVM, RF
- Collect metrics, save models and results

# setup search params for three types of models like so.
# they need the prefix__ because scaling is included in the pipelines
logreg_grid = {
    'logistic__penalty': ['l1', 'l2'],
    'logistic__C': np.logspace(-4, 4, 20),
    'logistic__solver' : ['liblinear']
    }
svm_grid = {
    'svm__C': np.logspace(-2, 4, 10),
    'svm__gamma': np.logspace(-3, 3, 10)
    }
rf_grid = {
    'rf__max_features': ['auto', 'sqrt'],
    'rf__n_estimators': [1, 20, 50, 100, 250],
    'rf__max_depth': list([10, 50, 100]).append(None),
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4, 6, 8,10]
    }
    
# Minimum number of samples required to split a node
# Minimum number of samples required at each leaf node
"""


print("loading packages...\n")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Save models w pickle
import pickle 
# parts of the pipeline for model selection
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import make_scorer, roc_auc_score, confusion_matrix, classification_report
# selected models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier



# Load data
print("\nLoading data...\n")
train_position = pd.read_csv("data/train_position.csv", index_col=0)
train_dummies = pd.read_csv("data/train_dummies.csv", index_col=0)
train_one_hot = pd.read_csv("data/train_one_hot.csv", index_col=0)
train_kmers = pd.read_csv("data/train_kmers.csv", index_col=0)

train_X = pd.concat([train_position, train_dummies, train_one_hot, train_kmers], 1) 
print("Training X data info:")
print(train_X.info())

train_Y = np.ravel(pd.read_csv("data/train_Y.csv", index_col=0))
print("Training Y shape:")
print(train_Y.shape)


test_position = pd.read_csv('data/test_position.csv')
test_dummies = pd.read_csv('data/test_dummies.csv')
test_one_hot = pd.read_csv('data/test_one_hot.csv')
test_kmers = pd.read_csv('data/test_kmers.csv')
test_X = pd.concat([test_position, test_dummies, test_one_hot, test_kmers], 1)

## Splitting data
print("\n\nSplitting data...\n")
# Split train/validate datasets - set test_size to 0.5 once ready
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=1)
sss.get_n_splits(train_X, train_Y)

for train_index, validate_index in sss.split(train_X, train_Y):
    x_train, x_validate = train_X.iloc[train_index, ], train_X.iloc[validate_index, ]
    y_train, y_validate = train_Y[train_index], train_Y[validate_index]

# X df's into numpy matrices, y's into 1d vectors 
x_train = x_train.to_numpy()
x_validate = x_validate.to_numpy()
y_train = np.ravel(y_train)
y_validate = np.ravel(y_validate)

x_train.shape
y_train.shape
x_validate.shape
y_validate.shape

del(train_X, train_Y, sss, train_index, validate_index)

### Setup model selection pipelines

# set up scaler to fit inside cv
scaler = StandardScaler()
# roc_auc scoring for model selection
auc_scoring = make_scorer(metrics.roc_auc_score) 


### Baseline model
x_scaled = scaler.fit_transform(x_train)
score = cross_val_score(
    LogisticRegression(random_state=0), 
    x_scaled, y_train, 
    cv=10, scoring = auc_scoring)
print("vanilla logistic regression 10-fold cv score")
print(score.mean())


basic_model = LogisticRegression(max_iter=10000)
basic_model.fit(x_train, y_train)
probs = basic_model.predict_proba(x_validate)

roc_auc_score(y_validate, probs[:, 1])
### Logistic Regression ##########

print('\n\n-----------------------------------')
print('Logistic regression search...\n')

# set classifier 
logistic = LogisticRegression(random_state = 0)
# setup steps to execute in nested cv pipeline
logreg_steps = [('scaler', scaler), ('logistic', logistic)]
logreg_pipe = Pipeline(steps = logreg_steps)

# Hyperparameters to tune for logistic regression
# (name__parameter tags need to match the estimators they're for)
logreg_grid = {
    'logistic__penalty': ['l1', 'l2'],
    'logistic__C': np.logspace(-4, 4, 20),
    'logistic__solver' : ['liblinear']
    }

# Setup the grid search
logreg_search = GridSearchCV(
    logreg_pipe, 
    logreg_grid, 
    scoring = auc_scoring,
    cv = 5,
    n_jobs = -1, 
    verbose = 3 
    )

# Now actually do the tuning search by CV
logreg_search.fit(x_train, y_train)
# save the model to disk
pickle.dump(logreg_search, open('logreg_search.sav', 'wb'))

# save search results
print("Best score (ROC AUC=%0.3f):" % logreg_search.best_score_)
print("Best parameters found for logistic regression:")
print(logreg_search.best_params_)
logreg_cv_results = pd.DataFrame(logreg_search.cv_results_)
logreg_cv_results.to_csv("./results/logreg_cv_results.csv")

# 'logistic__solver': 'liblinear', 'logistic__penalty': 'l1', 'logistic__C': 0.03359818286283781

# predict validation set using best logisitic regression model
# model is already fit on full training data by default from RandomizedSearchCV
logreg_y_pred = logreg_search.predict(x_validate)
logreg_y_prob = logreg_search.predict_proba(x_validate)
print(classification_report(y_validate, logreg_y_pred))
print(confusion_matrix(y_validate, logreg_y_pred, labels=[0,1]))

# collect key metrics for best logistic regression
logreg_perf = pd.DataFrame()
logreg_perf['algortithm'] = ['Logistic regression']
logreg_perf['roc_auc'] = [metrics.roc_auc_score(y_validate, logreg_y_prob[:, 1])]
logreg_perf['accuracy'] = [metrics.accuracy_score(y_validate, logreg_y_pred)]
logreg_perf['precision'] = [metrics.precision_score(y_validate, logreg_y_pred)]
logreg_perf['recall'] = [metrics.recall_score(y_validate, logreg_y_pred)]
logreg_perf['F1'] = [metrics.f1_score(y_validate, logreg_y_pred)]

print("\nTest performance of Logistic regression model")
print(logreg_perf)


### SVM ##########

print('\n\n-----------------------------------')
print('SVM search....\n')

# (same workflow as explained for logistic regression)
# tune the C and gamma parameters for RBF kernel
scaler = StandardScaler()
svm = SVC(kernel = 'rbf', random_state = 0)
svm_pipe = Pipeline(steps = [('scaler', scaler), ('svm', svm)])
# setup search params
svm_grid = {
    'svm__C': np.logspace(-2, 4, 10),
    'svm__gamma': np.logspace(-3, 3, 10)
    }
# perform search
svm_search = RandomizedSearchCV(
    svm_pipe, 
    svm_grid, 
    n_iter = 40, 
    cv = 5,
    scoring = auc_scoring,
    n_jobs = -1, 
    verbose = 3
    )
svm_search.fit(x_train, y_train)

# save the model to disk
pickle.dump(svm_search, open('svm_search.sav', 'wb'))

# save cv results
print("Best SVM score (CV score=%0.3f):" % svm_search.best_score_)
print("Best SVM parameters:")
print(svm_search.best_params_)
svm_cv_results = pd.DataFrame(logreg_search.cv_results_)
svm_cv_results.to_csv("./results/svm_cv_results.csv")

# predict validation set using best svm
svm_y_pred = svm_search.predict(x_validate)
svm_y_prob = svm_search.predict_proba(x_validate)
print(classification_report(y_validate, svm_y_pred))
print(confusion_matrix(y_validate, svm_y_pred, labels=[0,1]))

# collect key metrics for best svm
svm_perf = pd.DataFrame()
svm_perf['algortithm'] = ['SVM']
svm_perf['roc_auc'] = [metrics.roc_auc_score(y_validate, svm_y_pred)]
svm_perf['accuracy'] = [metrics.accuracy_score(y_validate, svm_y_pred)]
svm_perf['precision'] = [metrics.precision_score(y_validate, svm_y_pred)]
svm_perf['recall'] = [metrics.recall_score(y_validate, svm_y_pred)]
svm_perf['F1'] = [metrics.f1_score(y_validate, svm_y_pred)]

print("Test performance of SVM model")
print(svm_perf)


### RF ##########

print('\n\n-----------------------------------')
print('RandomForest search\n')

# Random forest models (same workflow as above)
rf = RandomForestClassifier(random_state = 0)
rf_grid = {
    'rf__max_features': ['auto', 'sqrt'],
    'rf__min_samples_leaf': [10, 50, 100, 200, 500],
    'rf__n_estimators': [100, 200, 500],
    'rf__bootstrap': [True]
    }
# 'rf__min_samples_split': [10],

rf_pipe = Pipeline(steps = [
    ('scaler', scaler), 
    ('rf', rf)]
    )
rf_search = RandomizedSearchCV(
    rf_pipe, 
    rf_grid,
    n_iter = 40, 
    cv = 5,
    scoring = auc_scoring,
    n_jobs=-1,
    verbose=3
    )
rf_search.fit(x_train, y_train)
# save the model to disk
pickle.dump(rf_search, open('rf_search.sav', 'wb'))

# best rf results
print("Best RF score (CV score=%0.3f):" % rf_search.best_score_)
print("Best RF params")
print(rf_search.best_params_)

# save cv results
rf_cv_results = pd.DataFrame(rf_search.cv_results_)
rf_cv_results.to_csv("./results/rf_cv_results.csv")

# predict validation set using best rf
rf_y_pred = rf_search.predict(x_validate)
print(classification_report(y_validate, rf_y_pred))
print(confusion_matrix(y_validate, rf_y_pred, labels=[0,1]))

# collect key metrics for best random forest
rf_perf = pd.DataFrame()
rf_perf['algortithm'] = ['Random forest']
rf_perf['roc_auc'] = [metrics.roc_auc_score(y_validate, rf_y_pred)]
rf_perf['accuracy'] = [metrics.accuracy_score(y_validate, rf_y_pred)]
rf_perf['precision'] = [metrics.precision_score(y_validate, rf_y_pred)]
rf_perf['recall'] = [metrics.recall_score(y_validate, rf_y_pred)]
rf_perf['F1'] = [metrics.f1_score(y_validate, rf_y_pred)]

print("Test performance of Random forest model")
print(rf_perf)

# # plot for variable importance
# importances = rf_search.best_estimator_._final_estimator.feature_importances_
# indices = np.argsort(importances)
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [i for i in indices])
# plt.xlabel('Relative Importance')
# plt.savefig('rf_importance_fig', bbox_inches="tight")
# plt.close()


# create a dataframe with performance metrics of best models
best_models = pd.concat([logreg_perf, svm_perf, rf_perf])
print(best_models)
best_models.to_csv("./results/test_performance.csv")





### SVM2 ##########

print('\n\n-----------------------------------')
print('SVM2 search....\n')

# (same workflow as explained for logistic regression)
# tune the C and gamma parameters for RBF kernel
scaler = StandardScaler()
svm = SVC(kernel = 'rbf', random_state = 0)
svm_pipe = Pipeline(steps = [('scaler', scaler), ('svm', svm)])
# setup search params
svm_grid = {
    'svm__C': np.logspace(-1, 1, 5),
    'svm__gamma': np.logspace(-5, -2, 5)
    }
# perform search
svm_search2 = GridSearchCV(
    svm_pipe, 
    svm_grid, 
    cv = 5,
    scoring = auc_scoring,
    n_jobs = -1, 
    verbose = 3
    )
svm_search2.fit(x_train, y_train)

# save the model to disk
pickle.dump(svm_search2, open('svm_search2.sav', 'wb'))

# save cv results
print("Best SVM2 score (CV score=%0.3f):" % svm_search2.best_score_)
print("Best SVM2 parameters:")
print(svm_search.best_params_)
svm2_cv_results = pd.DataFrame(svm_search2.cv_results_)
svm2_cv_results.to_csv("./results/svm2_cv_results.csv")

# predict validation set using best svm
svm2_y_pred = svm_search.predict(x_validate)
print(classification_report(y_validate, svm2_y_pred))
print(confusion_matrix(y_validate, svm2_y_pred, labels=[0,1]))

# collect key metrics for best svm
svm2_perf = pd.DataFrame()
svm2_perf['algortithm'] = ['SVM2']
svm2_perf['roc_auc'] = [metrics.roc_auc_score(y_validate, svm2_y_pred)]
svm2_perf['accuracy'] = [metrics.accuracy_score(y_validate, svm2_y_pred)]
svm2_perf['precision'] = [metrics.precision_score(y_validate, svm2_y_pred)]
svm2_perf['recall'] = [metrics.recall_score(y_validate, svm2_y_pred)]
svm2_perf['F1'] = [metrics.f1_score(y_validate, svm2_y_pred)]

print("Test performance of SVM2 model")
print(svm2_perf)




### RF ##########

print('\n\n-----------------------------------')
print('RandomForest search\n')

# Random forest models (same workflow as above)
rf2 = RandomForestClassifier(random_state = 0)
rf2_grid = {
    'rf__max_features': ['auto', 'sqrt'],
    'rf__n_estimators': [10, 20, 50, 100, 250],
    'rf__max_depth': list([10, 50, 100, 999999]),
    'rf__min_samples_split': [2, 5, 10, 20],
    'rf__min_samples_leaf': [1, 2, 5, 10, 20]
    }
# 'rf__min_samples_split': [10],

rf2_pipe = Pipeline(steps = [
    ('scaler', scaler), 
    ('rf', rf2)]
    )
rf2_search = RandomizedSearchCV(
    rf2_pipe, 
    rf2_grid,
    n_iter = 100, 
    cv = 5,
    scoring = auc_scoring,
    n_jobs=-1,
    verbose=3
    )
rf2_search.fit(x_train, y_train)
# save the model to disk
pickle.dump(rf2_search, open('rf2_search.sav', 'wb'))

# best rf results
print("Best RF2 score (CV score=%0.3f):" % rf2_search.best_score_)
print("Best RF2 params")
print(rf2_search.best_params_)

# save cv results
rf2_cv_results = pd.DataFrame(rf2_search.cv_results_)
rf2_cv_results.to_csv("./results/rf2_cv_results.csv")

# predict validation set using best rf
rf2_y_pred = rf2_search.predict(x_validate)
print(classification_report(y_validate, rf2_y_pred))
print(confusion_matrix(y_validate, rf2_y_pred, labels=[0,1]))

# collect key metrics for best random forest
rf2_perf = pd.DataFrame()
rf2_perf['algortithm'] = ['Random forest 2']
rf2_perf['roc_auc'] = [metrics.roc_auc_score(y_validate, rf2_y_pred)]
rf2_perf['accuracy'] = [metrics.accuracy_score(y_validate, rf2_y_pred)]
rf2_perf['precision'] = [metrics.precision_score(y_validate, rf2_y_pred)]
rf2_perf['recall'] = [metrics.recall_score(y_validate, rf2_y_pred)]
rf2_perf['F1'] = [metrics.f1_score(y_validate, rf2_y_pred)]

print("Test performance of Random forest model")
print(rf2_perf)


best_models = pd.concat([best_models, svm2_perf, rf2_perf], 0)
print(best_models)
best_models.to_csv("./results/test_performance.csv")






### svm3 ##########

print('\n\n-----------------------------------')
print('svm3 search....\n')

# (same workflow as explained for logistic regression)
# tune the C and gamma parameters for RBF kernel
scaler = StandardScaler()
svm = SVC(kernel = 'rbf', random_state = 0)
svm_pipe = Pipeline(steps = [('scaler', scaler), ('svm', svm)])
# setup search params
svm_grid = {
    'svm__C': np.logspace(-2, 2, 10),
    'svm__gamma': np.logspace(-5, -2, 8)
    }
# perform search
svm_search3 = GridSearchCV(
    svm_pipe, 
    svm_grid, 
    cv = 5,
    scoring = auc_scoring,
    n_jobs = 8, 
    verbose = 3
    )
svm_search3.fit(x_train, y_train)

# save the model to disk
pickle.dump(svm_search3, open('svm_search3.sav', 'wb'))

# save cv results
print("Best svm3 score (CV score=%0.3f):" % svm_search3.best_score_)
print("Best svm3 parameters:")
print(svm_search3.best_params_)
svm3_cv_results = pd.DataFrame(svm_search3.cv_results_)
svm3_cv_results.to_csv("./results/svm3_cv_results.csv")

# predict validation set using best svm
svm3_y_pred = svm_search3.predict(x_validate)
print(classification_report(y_validate, svm3_y_pred))
print(confusion_matrix(y_validate, svm3_y_pred, labels=[0,1]))

# collect key metrics for best svm
svm3_perf = pd.DataFrame()
svm3_perf['algortithm'] = ['SVM3']
svm3_perf['roc_auc'] = [metrics.roc_auc_score(y_validate, svm3_y_pred)]
svm3_perf['accuracy'] = [metrics.accuracy_score(y_validate, svm3_y_pred)]
svm3_perf['precision'] = [metrics.precision_score(y_validate, svm3_y_pred)]
svm3_perf['recall'] = [metrics.recall_score(y_validate, svm3_y_pred)]
svm3_perf['F1'] = [metrics.f1_score(y_validate, svm3_y_pred)]

print("Test performance of svm3 model")
print(svm3_perf)

