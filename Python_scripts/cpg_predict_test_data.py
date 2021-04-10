import numpy as np
import pandas as pd
import pickle 
# parts of the pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# what were best hyperparameters?


# load the choosen model
with open("./models/logreg_search.sav", 'rb') as pickle_file:
    logreg_model = pickle.load(pickle_file)
# warnings because sci-kit learn versions were different

logreg_model.best_estimator_
logreg_model.best_score_
logreg_model.best_params_

test_X = pd.read_csv('data/test_X.csv',  index_col=0)
test_X.shape
test_X = test_X.to_numpy() 

predictions = logreg_model.predict(test_X)
np.savetxt("predicted_test_values.csv", predictions, delimiter=",")
pd.DataFrame(predictions).to_csv("./results/Test_Predictions.csv",  header = ["Beta"], index=None)

