## Do model selection by 10-fold CV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# parts of the pipeline for model selection
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
# selected models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# reporting
from sklearn.metrics import confusion_matrix, classification_report


## Splitting data

# Load data
train_X = pd.read_csv("data/train_X.csv", index_col=0)
train_Y = pd.read_csv("data/train_Y.csv", index_col=0).to_numpy()

train_X.info()
train_Y.shape

# Split train/validate datasets - set test_size to 0.25 once ready
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=1)
sss.get_n_splits(train_X, train_Y)

for train_index, validate_index in sss.split(train_X, train_Y):
    x_train, x_validate = train_X.iloc[train_index, ], train_X.iloc[validate_index, ]
    y_train, y_validate = train_Y[train_index], train_Y[validate_index]

del(train_X, train_Y, sss, train_index, validate_index)

# X df's into numpy matrices, y's into 1d vectors 
x_train = x_train.to_numpy()
x_validate = x_validate.to_numpy()
y_train = np.ravel(y_train)
y_validate = np.ravel(y_validate)

x_train.shape
y_train.shape
x_validate.shape
y_validate.shape


## Setup model selection pipelines

# I prefer to use roc_auc as the metric, instead of accuracy
auc_scorer = make_scorer(roc_auc_score)
metrics.roc_auc_score

# set up preprocessing steps to do in nested cv
scaler = StandardScaler()
pca = PCA()


### Setup logistic regression tuning search

# set classifier 
logistic = LogisticRegression(max_iter=1000, tol=0.1, random_state = 22)
# setup steps to execute in nested cv pipeline
logreg_steps = [('scaler', StandardScaler()), ('pca', PCA()), ('logistic', logistic)]
logreg_pipe = Pipeline(steps = logreg_steps)
logreg_pipe

# Hyperparameters to tune for logistic regression
# (name__parameter tags need to match the estimators they're for)
logreg_grid = {
    'pca__n_components': [5, 10, 20, 50, 100, 250],
    'logistic__penalty': ['l1', 'l2'],
    'logistic__C': np.logspace(-4, 4, 20),
    'logistic__solver' : ['liblinear']
    }

# Setup the grid search
logreg_model = RandomizedSearchCV(
    logreg_pipe, 
    logreg_grid, 
    n_jobs=-1, 
    verbose=2, 
    n_iter = 40, 
    cv=10)

# Now actually do the tuning search by CV
print('\n\n-----------------------------------')
print('Logistic regression grid search\n')
logreg_model.fit(x_train, y_train)

print("Best parameter (CV score=%0.3f):" % logreg_model.best_score_)
print(logreg_model.best_params_)

# save search results
logreg_cv_results = pd.DataFrame(logreg_model.cv_results_)
logreg_cv_results.to_csv("./logreg_cv_results.csv")

# predict validation set using best logisitic regression model
logreg_y_pred = logreg_model.predict(x_validate)
print(classification_report(y_validate, logreg_y_pred))
print(confusion_matrix(y_validate, logreg_y_pred, labels=[0,1]))



# Plot the PCA spectrum
pca.fit(x_train)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(np.arange(1, pca.n_components_ + 1),
         pca.explained_variance_ratio_, '+', linewidth=2)
ax0.set_ylabel('PCA explained variance ratio')
ax0.axvline(logreg_model.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))

# For each number of components, find the best classifier results
components_col = 'param_pca__n_components'
best_clfs = logreg_cv_results.groupby(components_col).apply(
    lambda g: g.nlargest(1, 'mean_test_score'))
best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
               legend=False, ax=ax1)
ax1.set_ylabel('Classification accuracy (val)')
ax1.set_xlabel('n_components')

plt.xlim(-1, 105)
plt.tight_layout()
plt.title('PCA of x_train')
fig.savefig('pca_fig', bbox_inches="tight")


### Support vector machines models 
print('\n\n-----------------------------------')
print('SVM search\n')

# (same workflow as explained for logistic regression)
# tune the C and gamma parameters for RBF kernel
svm = SVC(kernel = 'rbf', random_state = 22)
svm_grid = {
    'pca__n_components': [5, 10, 20, 50, 100, 250],
    'svm__C': np.logspace(-2, 10, 13),
    'svm__gamma': np.logspace(-9, 3, 13)
    }
svm_pipe = Pipeline(steps = [
    ('scaler', StandardScaler()), 
    ('pca', PCA()), 
    ('svm', svm)]
    )
svm_model = RandomizedSearchCV(svm_pipe, svm_grid, n_jobs=-1, verbose=2, n_iter = 10, cv=10)
svm_model.fit(x_train, y_train)
svm_cv_results = pd.DataFrame(logreg_model.cv_results_)
svm_cv_results.to_csv("./svm_cv_results.csv")

###

#### EVERYTHING ABOVE HERE WORKS

### RF
print('\n\n-----------------------------------')
print('RandomForest search\n')

# Random forest models (same workflow as above)
rf = RandomForestClassifier(random_state = 22)
rf_grid = {
    'pca__n_components': [5, 10, 20, 50, 100, 250],
    'rf__max_features': ['auto', 'sqrt'],
    'rf__min_samples_leaf': [10, 50, 100],
    'rf__min_samples_split': [10],
    'rf__n_estimators': [1000],
    'rf__bootstrap': [True, False]
    }
rf_pipe = Pipeline(steps = [
    ('scaler', StandardScaler()), 
    ('pca', PCA()), 
    ('rf', rf)]
    )
rf_model = RandomizedSearchCV(
    rf_pipe, 
    rf_grid,
    n_iter = 10,
    cv=10,
    n_jobs=-1,
    verbose=2
    )
rf_model.fit(x_train, y_train)

rf_cv_results = pd.DataFrame(rf_model.cv_results_)
rf_cv_results.to_csv("./rf_cv_results.csv")

importances = rf_model.feature_importances_
importances.to_csv("./rf_feature_importances.csv")

indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Save Models Using Pickle
import pickle

# save the models to disk

pickle.dump(model, open('logreg_model.sav', 'wb'))
pickle.dump(model, open('svm_model.sav', 'wb'))
pickle.dump(model, open('rf_model.sav', 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)




# Plot the PCA spectrum
pca.fit(x_train)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(np.arange(1, pca.n_components_ + 1),
         pca.explained_variance_ratio_, '+', linewidth=2)
ax0.set_ylabel('PCA explained variance ratio')
ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))

# For each number of components, find the best classifier results
results = pd.DataFrame(search.cv_results_)
components_col = 'param_pca__n_components'
best_clfs = results.groupby(components_col).apply(
    lambda g: g.nlargest(1, 'mean_test_score'))
best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
               legend=False, ax=ax1)
ax1.set_ylabel('Classification accuracy (val)')
ax1.set_xlabel('n_components')

# plt.xlim(-1, 105)
plt.tight_layout()
plt.title('PCA of x_train', y=1.1)
fig.savefig('pca_fig', bbox_inches="tight")

