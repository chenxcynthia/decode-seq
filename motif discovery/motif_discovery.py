
# coding: utf-8

# # Model comparison (version 1)

# ### Load required packages and functions
# - Tensorflow package
# - Sklearn package
# - Functions in tcga_bcr.py
# 

# In[349]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from scipy.stats.mstats import zscore
from matplotlib.pyplot import colorbar

from tcga_bcr import *
from tcga_bcr_comp import plot_figures, plot_roc
from deep_bcr import *
from seq_logo_functions import *


# ### All functions
# 
# 1. Functions for processing and preparing kmer data
# 2. Clustering function(s)
# 3. Functions for calculating + visualizing PWMs
# 4. Functions for calculating + visualizing correlation matrices

# In[369]:

# Functions for processing and preparing kmer data
def binary_to_kmer(binary, aa_list='ACDEFGHIKLMNPQRSTVWY'):
    kmer = ''
    for i in range(len(binary)):
        for j in range(len(aa_list)):
            if(binary[i][j] == 1):
                kmer += aa_list[j]
    return kmer

def kmer_to_binary(kmer, aa_list='ACDEFGHIKLMNPQRSTVWY'):
    binary = []
    for i in range(len(kmer)):
        char = kmer[i]
        index = aa_list.find(char)
        array = np.zeros(len(aa_list))
        array[index] = 1
        binary.append(array)
    return binary

def generate_rand_kmer(kmer_size, aa_list='ACDEFGHIKLMNPQRSTVWY'):
    kmer = []
    letter = ''
    for i in range(kmer_size):
        binary = np.zeros(20);
        rand_index = np.random.randint(20);
        binary[rand_index] = 1
        kmer.append(binary)
        letter += aa_list[rand_index]
    #print('Positive kmers:', letter)
    return kmer, letter

# Clustering function
def kmer_similarity(kmer1, kmer2):
    score = 0
    for i in range(len(kmer1)):
        if kmer1[i] == kmer2[i]:
            score += 1
    return score

# Functions for calculating + visualizing PWMs
def get_weights(cancer_type):
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    weights = np.zeros((num_models, len(aa_list), kmer_size))

    for m in range(num_models):
        prediction = np.asarray(high_predictions_ctype)[m][cancer_type]
        total_sum = 0
        for i in range(top):
            total_sum += float(prediction[i, 0])

        for i in range(len(aa_list)):
            for j in range(kmer_size):
                weight = 0
                for p in range(top):
                    if prediction[p, 1][j] == aa_list[i]:
                        weight += float(prediction[p, 0])
                weights[m, i, j] = int(round((weight * 100 / total_sum), 1) * 10)
                
    return weights

# Calculates PWM from a 2 x p array (p is the # of predictions)
def pwm_calc(prediction):
    k = len(prediction[0][1])
    weights = np.zeros((len(aa_list), k))

    total_sum = 0
    for i in range(len(prediction)):
        total_sum += float(prediction[i][0])

    for i in range(len(aa_list)):
        for j in range(k):
            weight = 0
            for p in range(len(prediction)):
                if prediction[p][1][j] == aa_list[i]:
                    weight += float(prediction[p][0])
            weights[i, j] = int(round((weight * 10 / total_sum), 1) * 10)
            
    return weights

# Plots a heatmap for the weights
def plot_weights(weights, plot_title):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(weights, cmap='Blues')

    # Set tick labels
    ax.set_xticks(np.arange(len(tumors)))
    ax.set_yticks(np.arange(len(tumors)))
    ax.set_xticklabels(tumors)
    ax.set_yticklabels(np.flip(tumors, axis = 0));

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar = plt.colorbar(im)

    # Turn spines off and create white grid
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(weights.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(weights.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(14) 
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14) 

    # set size of tick labels
    plt.tick_params(axis='both', which='major', labelsize=18)
    cbar.ax.tick_params(labelsize=18) 
    
    # set title
    #plt.title(plot_title, fontsize=20)
    
    from matplotlib import rcParams
    rcParams['axes.titlepad'] = 20 
  
    plt.ylabel('Model 1', fontsize = 20)
    plt.xlabel('Model 2', fontsize = 20)
    plt.show()
    
# Functions for calculating + visualizing correlation matrices
def correlation_matrix(corr, xlabel, ylabel):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('hot', 100)
    cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap)
    plt.title('Pearson Feature Correlation Matrix', fontsize = 20)
    #ax1.set_xticklabels(labels,fontsize=6)
    #ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)
    fig.colorbar(cax)
    plt.show()
    
def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))


# ### Load required data & train 2 models with different parameters
# - Meta data for TCGA samples
# - Case name
# - Tumor labels
# - Train respective models

# In[488]:

para1 = (200, 1000, 500, 6, 20) # parameters for model 1
para2 = (200, 1000, 1000, 6, 20) # parameters for model 1
num_models = 2 # number of different models
num_labels = len(tumors)
aa_list = 'ACDEFGHIKLMNPQRSTVWY'

t = 10000 # Number of random test cases
top = 100 # Number of top test cases to choose from each tumor type

thresh = 3 #threshold for cluster size


# In[489]:

case = 'Tophat_Seq30Min300'
para = para1

num_motifs, batch_size, max_num_kmer, kmer_size, encode_size = para
meta = read_meta('../data/TCGA/tcga_clinical_CC.csv')
data_info, train_data, test_data = load_data(meta, case)
tumors = data_info.tumors.tolist()
labels = setup_tumor_types(meta, tumors)
features = format_balance_labels
encode = encode_aa_seq_binary_ext_dim
count = encode_count_genes
model_path = os.path.join('../work/', case+'-%s-%s-%s-%s-%s'%para)
model1 = GeneSwitchModelFast(num_motifs=num_motifs, num_labels=len(tumors), encode_init=encode_size, 
                                   save_path=model_path)


# In[350]:

case = 'Tophat_Seq30Min300'
para = para2

num_motifs, batch_size, max_num_kmer, kmer_size, encode_size = para
#meta = read_meta('../data/TCGA/tcga_clinical_CC.csv') - don't have to load again
data_info, train_data, test_data = load_data(meta, case)
tumors = data_info.tumors.tolist()
labels = setup_tumor_types(meta, tumors)
features = format_balance_labels
encode = encode_aa_seq_binary_ext_dim
count = encode_count_genes
model_path = os.path.join('../work/', case+'-%s-%s-%s-%s-%s'%para)
model2 = GeneSwitchModelFast(num_motifs=num_motifs, num_labels=len(tumors), encode_init=encode_size, 
                                   save_path=model_path)


# In[415]:

encode_size


# In[413]:

len(train_data[0])


# In[422]:

data = next(get_next_batch(test_data, labels, size=None, kmer=kmer_size))
for i in range(1):
    x, CS, ys = format_trim_dims(data, dim=max_num_kmer, encode=encode, count=count)


# In[ ]:




# ### Testing
# 
# - Define functions
# - Generate random kmers and feed them into NN
# - Sort kmers by output prediction score

# In[363]:

# Generate random testing kmer data
x = []
for i in range(t):
    # generates random test kmer to be "positive"
    rand_kmer, s = np.asarray(generate_rand_kmer(kmer_size))
    x.append(rand_kmer)

# fits xs to valid input dimensions
x = np.expand_dims(np.asarray(x), axis = 1)
CS = np.ones((t, 1, 5))
ys = np.ones((t, num_labels))


# In[423]:

# Predictions
model1.load(x, CS, ys) #max_iterations=100
pred1 = model1.predict(x, CS)
model2.load(x, CS, ys)
pred2 = model2.predict(x, CS)


# In[362]:

#plt.hist(zscore(pred1[:, 0]), 200)


# In[490]:

pred1.shape


# In[371]:

# Computing tumor-specific prediction score
ctype_predictions = []
pred = [pred1, pred2]

for m in range(num_models):
    ctype_predictions.append([])
    for i in range(t):
        ctype_predictions[m].append([])
        for j in range(num_labels):
            ctype_predictions[m][i].append((sigmoid(pred[m][i][j]), binary_to_kmer(x[i][0])))


# In[355]:

#normalize??
#np.divide(arr, np.linalg.norm(arr))


# In[372]:

# Sorting each cancer type by prediction score
high_predictions_ctype = []
for m in range(num_models):
    high_predictions_ctype.append([])
    for i in range(num_labels):
        # Sorts kmers based on prediction score and selects top kmers
        predarray = np.asarray(ctype_predictions)[m,:,i]
        if len(predarray) == 0:
            continue
        ind = np.lexsort((predarray[:,1], predarray[:,0]))    
        a = predarray[ind]
        
        float_a = []
        for i in a[:, 0]:
            float_a.append(float(i))
        top = (zscore(float_a) > 1).sum() # uses z-score to compute threshold
        
        high_predictions = a[-top:, :] # selects topkmers for sequence logo
        high_predictions = high_predictions[::-1]
        high_predictions_ctype[m].append(high_predictions)


# In[ ]:

# for m in range(num_models):
#     for i in range(num_labels):
#         print('Cancer type: ' + tumors[i])
#         print('\n'.join(high_predictions_ctype[m][i][:, 1].tolist()) + '\n')


# ### Clustering on real data
# 
# For each cancer type, we take the top kmers and cluster them in order to determine the binding kmers. 

# In[468]:

# Forms clusters until there are no similar kmers
all_clusters2 = []

for i in range(num_models):
    all_clusters2.append([])
    for c in range(num_labels):
        cluster_pred = high_predictions_ctype[i][c]
        clusters = []
        while (len(cluster_pred)>0):
            cluster = []
            top_kmer = cluster_pred[0, 1]
            j = 0
            while j < len(cluster_pred):
                if kmer_similarity(top_kmer, cluster_pred[j, 1]) > 1:
                    cluster.append(np.asarray(cluster_pred[j]))
                    cluster_pred = np.delete(cluster_pred, j, 0)
                    j = j-1
                j += 1
            if len(cluster) > 1:
                clusters.append(np.asarray(cluster))
        all_clusters2[i].append(np.asarray(clusters))
all_clusters2 = np.asarray(all_clusters2)


# In[496]:

all_clusters2[0][2][2].shape


# In[501]:

len(high_predictions_ctype[0][0][0]) 2 x 13 x 1150 x 2


# In[ ]:

# binary encoding


# In[ ]:

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", affinity=metric)
model.fit(X)


# #### Visualizing cluster analytics using histograms
# - size of clusters
# - number of clusters

# In[487]:

# Calculation
cluster_sizes = []
num_clusters_hist = []
for i in range(num_models):
    cluster_sizes.append([])
    num_clusters_hist.append([])
    for j in range(num_labels):
        num_clusters_hist[i].append(len(all_clusters2[i][j]))
        for k in range(len(all_clusters2[i][j])):
            cluster_sizes[i].append(len(all_clusters2[i][j][k]))
            
# Visualization
fig = plt.figure(figsize = (15, 10))
for i in range(4):
    plt.subplot(2,2,i+1)
    if (i<2):
        plt.title('Number of clusters: Model ' + str(((i%2)+1)))
        plt.hist(num_clusters_hist[i%2]) #rwidth = 0.6
        plt.xlabel('Number of clusters')
        plt.ylabel('Frequency')
    else:
        plt.title('Cluster sizes: Model ' + str(((i%2)+1)))
        plt.hist(cluster_sizes[i%2])
        plt.xlabel('Cluster size')
        plt.ylabel('Frequency')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
import pylab as pl
pl.savefig('hist.pdf', format='pdf')


# ### PWM calculation & Pearson feature correlation
# - Define functions
# - Calculate PWM for both models (weight matrix for each model and each cancer type)
# - Compute pearson correlation matrix + visualize using heatmap

# Compute correlation matrix

# In[478]:

thresh = 2
corr = np.zeros((num_labels, num_labels))
for i1 in range(num_labels): #ctype model 1
    for i2 in range(num_labels): #ctype model 2
        pwm1 = []
        pwm2 = []
        #for c1 in range(len(all_clusters2[0][i1])): #cluster model 1
        for c1 in range(3):
            if (len(all_clusters2[0][i1][c1]) >= thresh):
                pwm1.append(pwm_calc(all_clusters2[0][i1][c1]).flatten())
        #for c2 in range(len(all_clusters2[1][i2])): #cluster model 2
        for c2 in range(3):
            if (len(all_clusters2[1][i2][c2]) >= thresh):
                pwm2.append(pwm_calc(all_clusters2[1][i2][c2]).flatten())
        if (len(pwm1) == 0 or len(pwm2) == 0):
            corr[i1][i2] = 0
        else:
            corr_matrix = corr2_coeff(np.asarray(pwm1), np.asarray(pwm2))
            corr[i1][i2] = np.amax(corr_matrix[0:np.asarray(pwm1).shape[0], -np.asarray(pwm2).shape[0]:])
        #print(np.corrcoef(cluster_model1, cluster_model2).shape)


# In[479]:

plt.imshow(corr, cmap=plt.get_cmap('Blues'))


# In[433]:

# for i in range(num_labels):
#     for j in range(num_labels):
#         total = 0
#         total += np.sum(corr[i])
#         total += np.sum(corr[:, j])
#         corr[i][j] = corr[i][j]/total


# In[493]:

# Try taking top 100 kmers - no clustering, just feature correlation.
# eh bad idea bc there is almost guaranteed 1
a = high_predictions_ctype[0][3][:, 1][0:100]
b = high_predictions_ctype[1][3][:, 1][0:100]
np.intersect1d(a,b)


# In[494]:

np.unique(a)


# In[480]:

from matplotlib.pyplot import colorbar
plt.imshow(corr, cmap=plt.get_cmap('Blues'))
plt.ylabel('Model 1')
plt.xlabel('Model 2')
plt.colorbar()


# In[381]:

thresh = 0
corr = np.zeros((num_labels, num_labels))
for i1 in range(num_labels): #ctype model 1
    for i2 in range(num_labels): #ctype model 2
        pwm1 = []
        pwm2 = []
        #for c1 in range(len(all_clusters2[0][i1])): #cluster model 1
        for c1 in range(5):
            if (len(all_clusters2[0][i1][c1]) >= thresh):
                pwm1.append(pwm_calc(all_clusters2[0][i1][c1]).flatten())
        #for c2 in range(len(all_clusters2[1][i2])): #cluster model 2
        for c2 in range(5):
            if (len(all_clusters2[1][i2][c2]) >= thresh):
                pwm2.append(pwm_calc(all_clusters2[1][i2][c2]).flatten())
        corr_matrix = corr2_coeff(np.asarray(pwm1), np.asarray(pwm2))
        corr[i1][i2] = np.amax(corr_matrix)


# In[382]:

plt.imshow(corr, cmap=plt.get_cmap('Blues'))
plt.ylabel('Model 1')
plt.xlabel('Model 2')
plt.colorbar()


# In[440]:

# Weighting
thresh = 0
corr = np.zeros((num_labels, num_labels))
for i1 in range(num_labels): #ctype model 1
    for i2 in range(num_labels): #ctype model 2
        pwm1 = []
        pwm2 = []
        #for c1 in range(len(all_clusters2[0][i1])): #cluster model 1
        for c1 in range(5):
            pwm1.append(np.multiply(pwm_calc(all_clusters2[0][i1][c1]), len(all_clusters2[0][i1][c1])).flatten())
        #for c2 in range(len(all_clusters2[1][i2])): #cluster model 2
        for c2 in range(5):
            pwm2.append(np.multiply(pwm_calc(all_clusters2[1][i2][c2]), len(all_clusters2[1][i2][c2])).flatten())
        if (len(pwm1) == 0 or len(pwm2) == 0):
            corr[i1][i2] = 0
        else:
            corr_matrix = np.corrcoef(pwm1, pwm2)
            corr[i1][i2] = np.amax(corr_matrix[0:np.asarray(pwm1).shape[0], -np.asarray(pwm2).shape[0]:])


# In[441]:

plt.imshow(pwm1)


# In[ ]:

plt.imshow(pwm2)


# In[442]:

plt.imshow(corr, cmap=plt.get_cmap('Blues'))
plt.ylabel('Model 1')
plt.xlabel('Model 2')
plt.colorbar()


# In[483]:

plot_weights(np.flip(corr, axis = 0), '')


# In[ ]:



