

import re
import numpy as np
import pandas as pd
import seaborn as sns
# import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# from sklearn.ensemble import RandomForestClassifier


### FEATURE ENGINEERING FUNCTIONS

def make_dummies(df):
    """Make various dummies for (relation to island), (refgene group),
    (regulatory features)"""

    dfx = df.copy()
    ## get dummies for "Relation_to_UCSC_CpG_Island": 5 levels
    dfx = pd.get_dummies(dfx, columns =['rel_to_island'], prefix_sep = '', prefix = '')
    
    ## pull terms from 'UCSC_RefGene_Group' lists into columns of counts
    for term in ["TSS200", "TSS1500", "Body", "5'UTR", "3'UTR", "1stExon"]:
        dfx[term] = dfx["refgene"].str.count(term)
        dfx[term] = dfx[term].fillna(0).astype('int32')

    ## create 2 sets of dummies from 'feature' (Regulatory_Feature_Group)
    df["cell_type_specific"] = df['feature'].str.count("_Cell_type_specific").fillna(0).astype('int32')
    for term in ["Gene_Associated", "NonGene_Associated", "Promoter_Associated", "Unclassified"]:
        dfx[term] = dfx['feature'].str.count(term).fillna(0).astype('int32')

    dfx = dfx.drop(columns = ['position', 'island', 'refgene', 'feature', 'fwd_seq', 'seq'])
    return(dfx)
  
  
  def make_relative_positions(df):
    """ island col has position info like: "chr1:2004858-2005346"
        We want to get the position of the CpG site relative to the start and stop of the island
        Many columns don't have island data so add a dummy to indicate whether it exists
    """
    dfx = df.copy()
    # dummy variable for whether has 'island' or NA
    dfx['has_island'] = np.where(dfx['island'].isna(), 0, 1)
    
    # postion of CpG relative to nearby island start position (lots of missing values though)
    dfx['isl_start'] = dfx['island'].str.extract(':(\d+)').astype('float64')
    dfx['dist_start'] = dfx['isl_start'] - dfx['position']
    dfx['dist_start'] = dfx['dist_start'].fillna(0)
    
    # same for distance to end of island
    dfx['isl_end'] = dfx['island'].str.extract('-(\d+)').astype('float64')
    dfx['dist_end'] = dfx['isl_end'] - df['position']
    dfx['dist_end'] = dfx['dist_end'].fillna(0)
    
    # return distance columns
    return(dfx[['has_island', 'dist_start', 'dist_end']])
  
  
  ## with help from: https://www.kaggle.com/thomasnelson/working-with-dna-sequence-data-for-ml

def make_kmer_freq(df):
    """returns vectorized kmer frequency features as dataframe"""
    
    def get_kmers(dna, k=6):
        """creates list of kmers from dna seq"""
        dna = dna.upper()
        kmers = [dna[x:x+k] for x in range(len(dna)+1-k)]
        kmers = ' '.join(kmers)
        return(kmers)
    
    # create new column of 
    mers = df.apply(lambda x: get_kmers(x['seq'], 6), axis = 1)
    tfidf = TfidfVectorizer() 
    X = tfidf.fit_transform(mers)
    kmers = tfidf.get_feature_names()
    kmer_df = pd.DataFrame(X.toarray(), columns=kmers)
    return(kmer_df)


def make_one_hot_seq(df):
    
    def one_hot_encode_dna(dna):
        """ One-hot encode a single DNA sequence: 
        Requires creating two encoders: LabelEncoder to get from string to numeric, then OneHotEncoder
        Converts DNA to numeric then to one-hot matrix with shape: len(dna)*4
        """
        # create label encoder for DNA symbols
        label_encoder = LabelEncoder() 
        label_encoder.fit(np.array(list('ACGTN')))

        # create one-hot encoder
        onehot_encoder = OneHotEncoder(sparse=False, dtype=int)

        # dna to numeric array
        dna = re.sub('[^ACGT]', 'N', dna.upper())
        dna = np.array(list(dna))
        dna_int = label_encoder.transform(dna) 
        dna_int = dna_int.reshape(len(dna_int), 1)

        # convert to one-hot
        dna_onehot = onehot_encoder.fit_transform(dna_int)
        return(dna_onehot)
    
    """
    Splits the region around CpG site in up + downstream
    Applies one-hot encoding to each sequence and returns 480 column df
    """
    dfx = df.copy()
    # split the upstream and downstream seq around '[CpG]'; rejoin the two halves
    dfx['fwd_seq_x'] = dfx['fwd_seq'].str.split('\[|\]', expand = True).apply(lambda x: x[0] + x[2], axis=1)

    # apply one_hot_encoding to sequence and flatten matrix into vector
    X1 = dfx.apply(lambda x: one_hot_encode_dna(x['fwd_seq_x']).flatten(), axis=1)
    
    # stack vectors into data frame with 480 columns (x1 to x480), since [120 bp * 4 bases] are encoded.
    X1 = pd.DataFrame(np.column_stack(list(zip(*X1))), 
                      columns = list(x+str(y) for y in range(120) for x in 'ACGT'))

    return(X1)


def make_all_features(df):
  """wrapper for all the above feature-generating functions"""
    return(pd.concat([make_relative_positions(df.copy()),
                      make_dummies(df.copy()), 
                      make_kmer_freq(df.copy()), 
                      make_one_hot_seq(df.copy())], 
                     1)
          )
  
  


### MAIN
  
# read training data
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
del(col)

train = train.drop(['id', 'chromosome', 'outcome'], 1)

Y = train['outcome']
Y.to_csv("data/train_Y.csv")

# make_dummies(train)
# make_kmer_freq(train)
# make_relative_positions(train)
# make_one_hot_seq(train)

X = make_all_features(train)
X.to_csv("data/train_X.csv")

## Same thing for test data
test = pd.read_csv('data/test.csv')
test = test.rename(columns={"Id": "id",
                              "CHR": "chromosome", 
                              "MAPINFO": "position",
                              "UCSC_CpG_Islands_Name": "island",  
                              "UCSC_RefGene_Group":"refgene",
                              "Relation_to_UCSC_CpG_Island": "rel_to_island",
                              "Regulatory_Feature_Group": "feature",
                              "Forward_Sequence":"fwd_seq"})

# change categorical variables dtypes
test['position'] = test['position'].astype('float64')
test["rel_to_island"] = test["rel_to_island"].astype("category")
for col in ["fwd_seq", "seq", "refgene"]:
    test[col] = test[col].astype("string")
del(col)

# drop unneeded data
test = test.drop(['id', 'chromosome'], 1)

# make_dummies(test)
# make_kmer_freq(test)
# make_relative_positions(test)
# make_one_hot_seq(test)

X_test = make_all_features(test)
X_test.to_csv("data/test_X.csv")


