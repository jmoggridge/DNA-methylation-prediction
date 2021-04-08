#!/usr/bin/python

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def make_dummies(df):
    """Make various dummies for (relation to island), (refgene group), (regulatory features)"""
    dfx = df.copy()
    
    # get dummies for "Relation_to_UCSC_CpG_Island": 5 levels
    dfx = pd.get_dummies(dfx, columns =['rel_to_island'], prefix_sep = '', prefix = '')
    
    # pull terms from 'UCSC_RefGene_Group' lists into columns of counts
    for term in ["TSS200", "TSS1500", "Body", "5'UTR", "3'UTR", "1stExon"]:
        dfx[term] = dfx["refgene"].str.count(term)
        dfx[term] = dfx[term].fillna(0).astype('int32')
    
    # create 2 sets of dummies from 'feature ~ Regulatory_Feature_Group
    dfx["cell_type_specific"] = df['feature'].str.count("_Cell_type_specific").fillna(0).astype('int32')
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
def make_kmer_freq(df, k):
    """returns vectorized kmer frequency features as dataframe"""
    
    def get_kmers(dna, k):
        """creates list of kmers from dna seq"""
        dna = dna.upper()
        kmers = [dna[x:x+k] for x in range(len(dna)+1-k)]
        kmers = ' '.join(kmers)
        return(kmers)
    
    # create new column of 
    mers = df.apply(lambda x: get_kmers(x['seq'], k), axis = 1)
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

#
# def make_all_features(df):
#     return(pd.concat([make_relative_positions(df.copy()), 
#                       make_dummies(df.copy()), 
#                       make_kmer_freq(df.copy()), 
#                       make_one_hot_seq(df.copy())], 1))
# 
################################################################

## Main ######

train = pd.read_csv('data/train.csv')

# give the variables shorter names
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

# change string variables dtypes
for col in ["fwd_seq", "seq", "refgene"]:
    train[col] = train[col].astype("str")

# position as float    
train['position'] = train['position'].astype('float64')
del(col)

# remove and save outcomes
Y = train['outcome']
Y.to_csv('data/train_Y.csv')

# drop unnecessary columns
train = train.drop(['id', 'chromosome', 'outcome'], 1)
print(train.info())

# make all the features and save data
print("Making training features ....")
train_position = make_relative_positions(train.copy())
print("Position df")
print(train_position.info())
train_position.to_csv('data/train_position.csv')

train_dummies = make_dummies(train.copy())
print("Dummies df")
print(train_dummies.info())
train_dummies.to_csv('data/train_dummies.csv')

train_one_hot = make_one_hot_seq(train.copy())
print("One hot 120bp sequence df")
print(train_one_hot.info())
train_one_hot.to_csv('data/train_one_hot.csv')

train_kmers = make_kmer_freq(train.copy(), 2) 
print("kmers df before removing kmers with n's")
print(train_kmers.shape)

train_kmers = train_kmers.loc[:,~train_kmers.columns.str.contains('n', case=False)] 
print("kmers df after removing kmers with n's")
print(train_kmers.shape)
print(train_kmers.info())
print(train_kmers.describe())
train_kmers.to_csv('data/train_kmers.csv')

print("Done with training data")

train_X = pd.concat([train_position, train_dummies, train_one_hot, train_kmers], 1) 
train_X_columns = list(train_X.columns.values)

# del(train, train_kmers, train_one_hot, train_dummies, train_position)

####################################################################

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
    test[col] = test[col].astype("str")

del(col)

# drop unneeded data
test = test.drop(['id', 'chromosome'], 1)

# make features
print("Making testing features ....")
# make all the features and save data
print("Making training features ....")
test_position = make_relative_positions(test.copy())
test_position.to_csv('data/test_position.csv')
print("test_position df")
print(test_position.info())

test_dummies = make_dummies(test.copy())
test_dummies.to_csv('data/test_dummies.csv')
print("test_dummies df")
print(test_dummies.info())

test_one_hot = make_one_hot_seq(test.copy())
test_one_hot.to_csv('data/test_one_hot.csv')
print("test_one_hot (120bp sequence) df")
print(test_one_hot.info())

test_kmers = make_kmer_freq(test.copy(), 2)
print("test_kmers df before removing kmers with n's")
print(test_kmers.shape)
test_kmers = test_kmers.loc[:,~test_kmers.columns.str.contains('n', case=False)] 
print("test_kmers df after removing kmers with n's")
print(test_kmers.shape)
print(test_kmers.info())
test_kmers.to_csv('data/test_kmers.csv')

# join test data features:
test_X = pd.concat([test_position, test_dummies, test_one_hot, test_kmers], 1)

# make sure test has the same columns as train
test_X = test_X[test_X.columns.intersection(train_X_columns)]
test_X.shape

# write to data/
test_X.to_csv('data/test_X.csv')
print("Done")


################################################################

# train_X = pd.concat([position, dummies, one_hot, kmers], 1)
# print(train_X.info())
# 
# print("Saving training features ....")
# train_X.to_csv("data/train_X.csv")

# del(train, train_X)

# test_X = pd.concat([test_position, test_dummies, test_one_hot, test_kmers], 1)
# print(test_X.info())
# 
# # write to file
# print("Saving testing features ...")
# test_X.to_csv("data/test_X.csv")

# del(test, test_X)


