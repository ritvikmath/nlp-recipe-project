
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

####################################
# Import Data
####################################

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

pickle_in = open("baking_data_title_ingredients.pickle", 'rb')
baking_data = pickle.load(pickle_in)

pickle_in = open("nutritional_info.pickle",'rb')
nutritional_df = pickle.load(pickle_in)


####################################
# Unpack list with the baking info
####################################
# Index nutritional_info
health_mask = baking_data[0].id.isin(nutritional_df.id)

# baking dataframe
df = baking_data[0][health_mask] #5000 obs
list(df.columns)

#nutrition df variables
list(nutritional_df.columns)


# Recipe IDs
baking_ids = df.id.values

# baking_strings - concatenated ingredients
baking_strings = np.array(baking_data[1])[health_mask].tolist() 


# list of strings representing each recipe (ingredients only)
# Remove Separaters
baking_strings = [item.replace('--|||--', '').replace('||', '') for item in np.array(baking_data[1])[health_mask].tolist()]
# Remove extra white space
baking_strings = [" ".join(item.split()) for item in baking_strings]

# In df, but not used:
# baking_tokens - instructions converted to token
# token_num_str = dictionary mapping words and tokens


# Concatenate strings and IDs; Take sample
baking_ids_with_strings = list(zip(baking_strings, baking_ids))
#shuffle(baking_ids_with_strings)
# only retain 5,000
#baking_ids_with_strings_limited = baking_ids_with_strings[:5000]
baking_sent = [instructions[0] for instructions in baking_ids_with_strings]
ids_limited= [id[1] for id in baking_ids_with_strings]


####################################
# Bag of Words Model
###################################

# Tokenization: Generate "Histogram"
wordfreq = {} #wordfreq - dictionary of "tokens' 67,858 for total baking_strings
for sentence in baking_sent:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

# Take 5,000 words that appear most often
#most_freq = heapq.nlargest(5000, wordfreq, key=wordfreq.get)

# Convert to feature vector -> feature matrix
sentence_vectors = []
for sentence in baking_sent:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in wordfreq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)

sentence_vectors = np.asarray(sentence_vectors)


# Compute cosine similarity
sim_matrix = 1-pairwise_distances(sentence_vectors, metric="cosine")
sim_matrix[sim_matrix == 1] = 0 # replace all perfect matches (match with itself) to zero
 
# Print most similar recipes for first 20 recipes
for index in range(20):
  print('Recipe:')
  print(df[df.id == ids_limited[index]].title.iloc[0])
  print('Similar Recipes:')
  row = sim_matrix[index]
  most_similar_rec = np.argpartition(row, -10)[-10:]
  most_similar_ids = [ids_limited[i] for i in most_similar_rec]
  print(df[df.id.isin(most_similar_ids)].title.values)
  print('--------------')


#########################################
## K- Means Clustering
#########################################

# K-Means clustering - perform on feature matrix (try to ID Savory vs. Sweet)
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(sentence_vectors)
clusters = km.labels_.tolist()

cluster_data = { 'recipe': ids_limited, 'cluster': clusters}
frame = pd.DataFrame(cluster_data, index = [clusters] , columns = ['recipe', 'cluster'])
frame['string'] = baking_strings
frame['cluster'].value_counts()

for i in range(num_clusters):
    recipe_ids = frame.recipe[i]
    print("Cluster %d Recipes:" % i)
    print(df[df.id.isin(recipe_ids)].title.values[0:25])
    
    
## Find keyword ingredients
# CLUSTER 0
wordfreq_cluster0 = {} 
cluster0_mask = frame['cluster']==0
cluster0_df = frame[cluster0_mask]
cluster0_baking_sent = cluster0_df['string'].tolist()
for sentence in cluster0_baking_sent:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_cluster0.keys():
            wordfreq_cluster0[token] = 1
        else:
            wordfreq_cluster0[token] += 1
most_freq_cluster0 = heapq.nlargest(50, wordfreq_cluster0, key=wordfreq_cluster0.get)            
print(most_freq_cluster0)        
            
# CLUSTER 1
wordfreq_cluster1 = {} 
cluster1_mask = frame['cluster']==1
cluster1_df = frame[cluster1_mask]
cluster1_baking_sent = cluster1_df['string'].tolist()
for sentence in cluster1_baking_sent:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq_cluster1.keys():
            wordfreq_cluster1[token] = 1
        else:
            wordfreq_cluster1[token] += 1
most_freq_cluster1 = heapq.nlargest(50, wordfreq_cluster1, key=wordfreq_cluster1.get)                    
print(most_freq_cluster1)


#########################################
## Write BoW to PICKLE
#########################################

pickle_in = open("model_similarity_info.p", 'rb')
model_data = pickle.load(pickle_in)
model_data["BoW"] = {'ids': frame['recipe'].tolist(), 'sim_mtx': sim_matrix}

pickle_out = open("model_similarity_info_full.pickle","wb")
pickle.dump(model_data,pickle_out)
pickle_out.close()

## Just BoW and embedding
model_data["BoW"] = {'ids': frame['recipe'].tolist(), 'sim_mtx': sim_matrix, 'embedding': sentence_vectors}
model_BoW = {'BoW': model_data["BoW"]}
pickle_out = open("BoW_info_full.pickle","wb")
pickle.dump(model_BoW,pickle_out)
pickle_out.close()

    
#########################################
## PCA + Kmeans
#########################################

### PCA
X_std = StandardScaler().fit_transform(sim_matrix)
pca = PCA(n_components=5).fit(X_std)
pca_2d = pca.transform(X_std) # PC data
plt.scatter(pca_2d[:,0], pca_2d[:,1], c =frame.cluster)


# Scree Plot: Variabiliy explained + visualization
pca.explained_variance_ratio_
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')

## K-means/ Determine number of clusters
PCA_components = pd.DataFrame(pca_2d)
ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:2])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

num_clusters = 2
km = KMeans(n_clusters=num_clusters)
km.fit(PCA_components)
clusters = km.labels_.tolist()

cluster_data = { 'recipe': ids_limited, 'cluster': clusters}
frame = pd.DataFrame(cluster_data, index = [clusters] , columns = ['recipe', 'cluster'])
frame['cluster'].value_counts()

for i in range(num_clusters):
    recipe_ids = frame.recipe[i]
    print("Cluster %d Recipes:" % i)
    print(df[df.id.isin(recipe_ids)].title.values[0:25])

plt.scatter(pca_2d[:,0], pca_2d[:,1], c =frame.cluster)


