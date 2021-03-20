# library(reticulate)
# py_install("sklearn", pip=TRUE)
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


train = pd.read_csv('data/train.csv')
train.shape
train.head
train.columns
train.info
train.describe(include='all')
train.describe(include='category')

x["UCSC_CpG_Islands_Name"]
print(list(zip(train.columns, train.dtypes)))
x["Regulatory_Feature_Group"]

train["Regulatory_Feature_Group"].count()

# make dummy variables for chromosome, UCSC_RefGene_Group, Regulatory_Feature_Group

train["MAPINFO"][1:10]

# split into folds/
k_folds = StratifiedKFold(n_splits=10)
k_folds

# 


