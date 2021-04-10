import numpy as np
import pandas as pd
import pickle 
# parts of the pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

scaler = StandardScaler()
scaler.fit(train_X)
X_scaled = pd.DataFrame(scaler.transform(train_X),columns = train_X.columns)

clf = LogisticRegression(random_state = 0, penalty = 'l1', C = 0.0336, solver = 'liblinear')
clf.fit(X_scaled, train_Y)

# variable importance
importance_df = pd.DataFrame({'feature': train_X.columns, 'coefficient': clf.coef_[0]})

importance_df.to_csv("./results/LR_variable_importance.csv") 
	
