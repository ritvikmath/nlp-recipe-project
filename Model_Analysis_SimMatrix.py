#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashleychiu
"""

import os
import pickle
import re
import nltk
import heapq
import numpy as np
from random import shuffle
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

os.getcwd()
os.chdir('/Users/ashleychiu/Documents/CS 263')

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            

####################################
# LOAD FULL BAKING DATA
####################################
pickle_in = open("baking_data_title_ingredients.pickle", 'rb')
baking_data = pickle.load(pickle_in)

pickle_in = open("nutritional_info.pickle",'rb')
nutritional_df = pickle.load(pickle_in)


# Index out
health_mask = baking_data[0].id.isin(nutritional_df.id)

# baking dataframe
df = baking_data[0][health_mask] #5000 obs
list(df.columns)

#nutrition df
list(nutritional_df.columns)


# Recipe IDs
baking_ids = df.id.values

baking_strings = np.array(baking_data[1])[health_mask].tolist()

# list of strings representing each recipe (instructions only)
# Remove Separaters
baking_strings = [item.replace('--|||--', '').replace('||', '') for item in np.array(baking_data[1])[health_mask].tolist()]
# Remove extra white space
baking_strings = [" ".join(item.split()) for item in baking_strings]  
            
baking_ids_with_strings = list(zip(baking_strings, baking_ids))

baking_sent = [instructions[0] for instructions in baking_ids_with_strings]
ids_limited= [id[1] for id in baking_ids_with_strings]


####################################
# LOAD APP DATA
####################################

pickle_in = open("model_similarity_info_full.pickle","rb")
model_data = pickle.load(pickle_in)

# model_data = [model][ids][sim_mtx] (nested dictionary)

# Make consistnet -- zero along diagonals
for i in range(model_data['BoW']['sim_mtx'].shape[1]):
    model_data['D2V']['sim_mtx'][i,i] = 0
    model_data['LSTM']['sim_mtx'][i,i] = 0


## Unpack dictionary
BoW_sim = model_data['BoW']['sim_mtx']
D2V_sim = model_data['D2V']['sim_mtx']
LSTM_sim = model_data['LSTM']['sim_mtx']
BERT_sim = model_data['BERT']['sim_mtx']

#######################
## INDEX 
recipe_index = [2,8,17]

#######################################################
## PRINT MOST SIMILAR RECIPES FROM EACH ###############
#######################################################

for index in recipe_index:
  print('Recipe:')
  print(df[df.id == ids_limited[index]].title.iloc[0])
  print('Similar Recipes:')
  row = BoW_sim[index]
  most_similar_rec = np.argpartition(row, -10)[-10:]
  most_similar_ids = [ids_limited[i] for i in most_similar_rec]
  print(df[df.id.isin(most_similar_ids)].title.values)
  print('--------------')
    
for index in recipe_index:
  print('Recipe:')
  print(df[df.id == ids_limited[index]].title.iloc[0])
  print('Similar Recipes:')
  row = D2V_sim[index]
  most_similar_rec = np.argpartition(row, -10)[-10:]
  most_similar_ids = [ids_limited[i] for i in most_similar_rec]
  print(df[df.id.isin(most_similar_ids)].title.values)
  print('--------------')


for index in recipe_index:
  print('Recipe:')
  print(df[df.id == ids_limited[index]].title.iloc[0])
  print('Similar Recipes:')
  row = LSTM_sim[index]
  most_similar_rec = np.argpartition(row, -10)[-10:]
  most_similar_ids = [ids_limited[i] for i in most_similar_rec]
  print(df[df.id.isin(most_similar_ids)].title.values)
  print('--------------')    
  
for index in recipe_index:
  print('Recipe:')
  print(df[df.id == ids_limited[index]].title.iloc[0])
  print('Similar Recipes:')
  row = BERT_sim[index]
  most_similar_rec = np.argpartition(row, -10)[-10:]
  most_similar_ids = [ids_limited[i] for i in most_similar_rec]
  print(df[df.id.isin(most_similar_ids)].title.values)
  print('--------------')    
  
  
#######################################################
## Cluster 
#######################################################  

num_clusters = 2
km = KMeans(n_clusters=num_clusters)

## BAG OF WORDS
km.fit(BoW_sim)
clusters = km.labels_.tolist()

cluster_data_BoW = { 'recipe': ids_limited, 'cluster': clusters}
frame_BoW = pd.DataFrame(cluster_data_BoW, index = [clusters] , columns = ['recipe', 'cluster'])
frame_BoW['string'] = baking_strings
frame_BoW['cluster'].value_counts()

for i in range(num_clusters):
    recipe_ids = frame_BoW.recipe[i]
    print("Cluster %d Recipes:" % i)
    print(df[df.id.isin(recipe_ids)].title.values[0:10])
        
## Find keyword ingredients
# CLUSTER 0
wordfreq_cluster0_BoW = {} 
cluster0_mask_BoW = frame_BoW['cluster']==0
cluster0_df_BoW = frame_BoW[cluster0_mask_BoW]
cluster0_baking_sent_BoW = cluster0_df_BoW['string'].tolist()
for sentence in cluster0_baking_sent_BoW:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_cluster0_BoW.keys():
            wordfreq_cluster0_BoW[token] = 1
        else:
            wordfreq_cluster0_BoW[token] += 1
most_freq_cluster0_BoW = heapq.nlargest(50, wordfreq_cluster0_BoW, key=wordfreq_cluster0_BoW.get)            
print(most_freq_cluster0_BoW)        
            
# CLUSTER 1
wordfreq_cluster1_BoW = {} 
cluster1_mask_BoW = frame_BoW['cluster']==1
cluster1_df_BoW = frame_BoW[cluster1_mask_BoW]
cluster1_baking_sent_BoW = cluster1_df_BoW['string'].tolist()
for sentence in cluster1_baking_sent_BoW:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_cluster1_BoW.keys():
            wordfreq_cluster1_BoW[token] = 1
        else:
            wordfreq_cluster1_BoW[token] += 1
most_freq_cluster1_BoW = heapq.nlargest(50, wordfreq_cluster1_BoW, key=wordfreq_cluster1_BoW.get)                    
print(most_freq_cluster1_BoW)

unique_BoW0 = set(most_freq_cluster0_BoW) - set(most_freq_cluster1_BoW)
print(unique_BoW0)
unique_BoW1 = set(most_freq_cluster1_BoW) - set(most_freq_cluster0_BoW)
print(unique_BoW1)


### PCA
X_std = StandardScaler().fit_transform(BoW_sim)
pca = PCA(n_components=5).fit(X_std)
pca_2d = pca.transform(X_std) # PC data
plt.scatter(pca_2d[:,0], pca_2d[:,1], c =frame_BoW.cluster)


pca.explained_variance_ratio_
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')

############################################
## DOC2VEC
km.fit(D2V_sim)
clusters = km.labels_.tolist()

cluster_data_D2V = { 'recipe': ids_limited, 'cluster': clusters}
frame_D2V = pd.DataFrame(cluster_data_D2V, index = [clusters] , columns = ['recipe', 'cluster'])
frame_D2V['string'] = baking_strings
frame_D2V['cluster'].value_counts()

for i in range(num_clusters):
    recipe_ids = frame_D2V.recipe[i]
    print("Cluster %d Recipes:" % i)
    print(df[df.id.isin(recipe_ids)].title.values[0:10])
        
## Find keyword ingredients
# CLUSTER 0
wordfreq_cluster0_D2V = {} 
cluster0_mask_D2V = frame_D2V['cluster']==0
cluster0_df_D2V = frame_D2V[cluster0_mask_D2V]
cluster0_baking_sent_D2V = cluster0_df_D2V['string'].tolist()
for sentence in cluster0_baking_sent_D2V:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_cluster0_D2V.keys():
            wordfreq_cluster0_D2V[token] = 1
        else:
            wordfreq_cluster0_D2V[token] += 1
most_freq_cluster0_D2V = heapq.nlargest(50, wordfreq_cluster0_D2V, key=wordfreq_cluster0_D2V.get)            
print(most_freq_cluster0_D2V)        
            
# CLUSTER 1
wordfreq_cluster1_D2V = {} 
cluster1_mask_D2V = frame_D2V['cluster']==1
cluster1_df_D2V = frame_D2V[cluster1_mask_D2V]
cluster1_baking_sent_D2V = cluster1_df_D2V['string'].tolist()
for sentence in cluster1_baking_sent_D2V:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_cluster1_D2V.keys():
            wordfreq_cluster1_D2V[token] = 1
        else:
            wordfreq_cluster1_D2V[token] += 1
most_freq_cluster1_D2V = heapq.nlargest(50, wordfreq_cluster1_D2V, key=wordfreq_cluster1_D2V.get)                    
print(most_freq_cluster1_D2V)

unique_D2V0 = set(most_freq_cluster0_D2V) - set(most_freq_cluster1_D2V)
print(unique_D2V0)
unique_D2V1 = set(most_freq_cluster1_D2V) - set(most_freq_cluster0_D2V)
print(unique_D2V1)

### PCA
X_std = StandardScaler().fit_transform(D2V_sim)
pca = PCA(n_components=5).fit(X_std)
pca_2d = pca.transform(X_std) # PC data
plt.scatter(pca_2d[:,0], pca_2d[:,1], c =frame_D2V.cluster)

pca.explained_variance_ratio_
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')


############################################
## LSTM
km.fit(LSTM_sim)
clusters = km.labels_.tolist()

cluster_data_LSTM = { 'recipe': ids_limited, 'cluster': clusters}
frame_LSTM = pd.DataFrame(cluster_data_LSTM, index = [clusters] , columns = ['recipe', 'cluster'])
frame_LSTM['string'] = baking_strings
frame_LSTM['cluster'].value_counts()

for i in range(num_clusters):
    recipe_ids = frame_LSTM.recipe[i]
    print("Cluster %d Recipes:" % i)
    print(df[df.id.isin(recipe_ids)].title.values[0:10])
        
## Find keyword ingredients
# CLUSTER 0
wordfreq_cluster0_LSTM = {} 
cluster0_mask_LSTM = frame_LSTM['cluster']==0
cluster0_df_LSTM = frame_LSTM[cluster0_mask_LSTM]
cluster0_baking_sent_LSTM = cluster0_df_LSTM['string'].tolist()
for sentence in cluster0_baking_sent_LSTM:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_cluster0_LSTM.keys():
            wordfreq_cluster0_LSTM[token] = 1
        else:
            wordfreq_cluster0_LSTM[token] += 1
most_freq_cluster0_LSTM = heapq.nlargest(50, wordfreq_cluster0_LSTM, key=wordfreq_cluster0_LSTM.get)            
print(most_freq_cluster0_LSTM)        
            
# CLUSTER 1
wordfreq_cluster1_LSTM = {} 
cluster1_mask_LSTM = frame_LSTM['cluster']==1
cluster1_df_LSTM = frame_LSTM[cluster1_mask_LSTM]
cluster1_baking_sent_LSTM = cluster1_df_LSTM['string'].tolist()
for sentence in cluster1_baking_sent_LSTM:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_cluster1_LSTM.keys():
            wordfreq_cluster1_LSTM[token] = 1
        else:
            wordfreq_cluster1_LSTM[token] += 1
most_freq_cluster1_LSTM = heapq.nlargest(50, wordfreq_cluster1_LSTM, key=wordfreq_cluster1_LSTM.get)                    
print(most_freq_cluster1_LSTM)

unique_LSTM0 = set(most_freq_cluster0_LSTM) - set(most_freq_cluster1_LSTM)
print(unique_LSTM0)
unique_LSTM1 = set(most_freq_cluster1_LSTM) - set(most_freq_cluster0_LSTM)
print(unique_LSTM1)

X_std = StandardScaler().fit_transform(LSTM_sim)
pca = PCA(n_components=5).fit(X_std)
pca_2d = pca.transform(X_std) # PC data
plt.scatter(pca_2d[:,0], pca_2d[:,1], c =frame_LSTM.cluster)

pca.explained_variance_ratio_
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')


############################################
## BERT
km.fit(BERT_sim)
clusters = km.labels_.tolist()

cluster_data_BERT = { 'recipe': ids_limited, 'cluster': clusters}
frame_BERT = pd.DataFrame(cluster_data_BERT, index = [clusters] , columns = ['recipe', 'cluster'])
frame_BERT['string'] = baking_strings
frame_BERT['cluster'].value_counts()

for i in range(num_clusters):
    recipe_ids = frame_BERT.recipe[i]
    print("Cluster %d Recipes:" % i)
    print(df[df.id.isin(recipe_ids)].title.values[0:10])
        
## Find keyword ingredients
# CLUSTER 0
wordfreq_cluster0_BERT = {} 
cluster0_mask_BERT = frame_BERT['cluster']==0
cluster0_df_BERT = frame_BERT[cluster0_mask_BERT]
cluster0_baking_sent_BERT = cluster0_df_BERT['string'].tolist()
for sentence in cluster0_baking_sent_BERT:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_cluster0_BERT.keys():
            wordfreq_cluster0_BERT[token] = 1
        else:
            wordfreq_cluster0_BERT[token] += 1
most_freq_cluster0_BERT = heapq.nlargest(50, wordfreq_cluster0_BERT, key=wordfreq_cluster0_BERT.get)            
print(most_freq_cluster0_BERT)        
            
# CLUSTER 1
wordfreq_cluster1_BERT = {} 
cluster1_mask_BERT = frame_BERT['cluster']==1
cluster1_df_BERT = frame_BERT[cluster1_mask_BERT]
cluster1_baking_sent_BERT = cluster1_df_BERT['string'].tolist()
for sentence in cluster1_baking_sent_BERT:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_cluster1_BERT.keys():
            wordfreq_cluster1_BERT[token] = 1
        else:
            wordfreq_cluster1_BERT[token] += 1
most_freq_cluster1_BERT = heapq.nlargest(50, wordfreq_cluster1_BERT, key=wordfreq_cluster1_BERT.get)                    
print(most_freq_cluster1_BERT)

unique_BERT0 = set(most_freq_cluster0_BERT) - set(most_freq_cluster1_BERT)
print(unique_BERT0)
unique_BERT1 = set(most_freq_cluster1_BERT) - set(most_freq_cluster0_BERT)
print(unique_BERT1)

X_std = StandardScaler().fit_transform(BERT_sim)
pca = PCA(n_components=5).fit(X_std)
pca_2d = pca.transform(X_std) # PC data
plt.scatter(pca_2d[:,0], pca_2d[:,1], c =frame_BERT.cluster)

pca.explained_variance_ratio_
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')