import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Load data
train_X = pd.read_csv("data/train_X.csv", index_col=0)
train_Y = pd.read_csv("data/train_Y.csv", index_col=0)

# train = pd.concat([train_X, train_Y], 1)
train_X.info()

 
train_X_scaled = StandardScaler().fit_transform(train_X)

pca = PCA(n_components=1000)
train_X_pca = pca.fit_transform(train_X_scaled)
print(pca.components_)
print(pca.explained_variance_)

pca_df = pd.DataFrame(train_X_pca, columns = ["PC1", "PC2", "PC3"])
pca_df['outcome'] = train_Y

plt.scatter(
  train_X_pca[:, 0], train_X_pca[:, 1], 
  alpha=0.5, 
  # c=np.ravel(train_Y),
  cmap=plt.cm.get_cmap('magma', 2)
  )
plt.axis('equal')
plt.xlabel('component 1')
plt.ylabel('component 2')

# def draw_vector(v0, v1, ax=None):
#     ax = ax or plt.gca()
#     arrowprops=dict(arrowstyle='->',
#                     linewidth=2,
#                     shrinkA=0, shrinkB=0)
#     ax.annotate('', v1, v0, arrowprops=arrowprops)
# 
# # plot data
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# 
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)
# plt.axis('equal');


pca_model.explained_variance_
pca_model.explained_variance_.cumsum()



# project test data
