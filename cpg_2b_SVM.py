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
svm_cv_results.to_csv("./svm_cv_results.csv")

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

