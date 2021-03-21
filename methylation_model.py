# library(reticulate)
# py_install("sklearn", pip=TRUE)
import re
import numpy as np
import pandas as pd
import seaborn as sns
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import MLPclassifier

train = pd.read_csv('data/train.csv')

# give the data names that don't suck
train = train.rename(columns={"Id": "id",
                              "CHR": "chromosome", 
                              "MAPINFO": "position",
                              "UCSC_CpG_Islands_Name": "island",  
                              "UCSC_RefGene_Group":"refgene",
                              "Relation_to_UCSC_CpG_Island": "rel_to_island",
                              "Regulatory_Feature_Group": "feature",
                              "Forward_Sequence":"fwd_seq",
                              "Beta": "outcome"})

# change categorical variables dtypes
for col in ["rel_to_island", "outcome"]:
    train[col] = train[col].astype("category")
for col in ["fwd_seq", "seq", "refgene"]:
    train[col] = train[col].astype("string")
train['position'] = train['position'].astype('float64')

y = train['outcome']
train = train.drop(['outcome'])
train.info()

def process_methylation_data(df):
  df = train.drop(['id', 'chromosome'], 1)
  
  # get dummies for "Relation_to_UCSC_CpG_Island": 5 levels
  df = pd.get_dummies(df, columns =['rel_to_island'], prefix_sep = '', prefix = '')

  # pull terms from 'UCSC_RefGene_Group' lists into columns of counts
  for term in ["TSS200", "TSS15000", "Body", "5'UTR", "3'UTR", "1stExon"]:
      df[term] = df["refgene"].str.count(term)
      df[term] = df[term].fillna(0).astype('int32')
  
  ## create 2 sets of dummies from 'feature' (Regulatory_Feature_Group)
  df["cell_type_specific"] = df['feature'].str.count("_Cell_type_specific").fillna(0).astype('int32')
  for term in ["Gene_Associated", "NonGene_Associated", "Promoter_Associated", "Unclassified"]:
      df[term] = df['feature'].str.count(term).fillna(0).astype('int32')
  
  ## postion of CpG relative to nearby island - lots of missing values though
  df['isl_start'] = df['island'].str.extract(':(\d+)').astype('float64')
  df['pos_to_start'] = df['position'] - df['isl_start']
  df['isl_end'] = df['island'].str.extract('-(\d+)').astype('float64')
  df['pos_to_end'] = df['isl_end'] - df['position']
  
  
  df = df.drop(columns = ['feature', 'isl_start', 'isl_end',
                          'position', 'island', 'refgene'])
  


#%%

# split into folds/
k_folds = StratifiedKFold(n_splits=10)
k_folds

# 


