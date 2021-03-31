## Do model selection by 10-fold CV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle # Save Models Using Pickle

# parts of the pipeline for model selection
from sklearn.decomposition import PCA
from sklearn import metrics
# selected models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# reporting
from sklearn.metrics import confusion_matrix, classification_report



probs = model.predict_proba(x_test)[:,1]
metrics.roc_auc_score(y_test, probs)


##############

## Plots

# cv search results for each model

# roc_auc curves
metrics.plot_roc_curve(logreg_model, x_validate, y_validate) 
metrics.plot_roc_curve(svm_model, x_validate, y_validate) 
metrics.plot_roc_curve(rf_model, x_validate, y_validate) 


# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)




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

