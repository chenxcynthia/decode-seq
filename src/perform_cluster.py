
# coding: utf-8

# # Single script for all model comparison

# ### Load required packages and functions
# - Tensorflow package
# - Sklearn package
# - Functions in tcga_bcr.py
# 

# In[1]:

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

# In[2]:

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

# Clustering functions
def compute_avg_max(array):
    total = 0
    array = np.asarray(array)
    for i in range(len(array)):
        total += np.max(array[i])
    for j in range(len(array[0])):
        total += np.max(array[:, j])
    return total/(len(array) + len(array[0]))

# Normalization functions

# Normalization by row and column
def norm_row_col(arr):
    norm = np.zeros((len(arr), len(arr[0])))
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            total = 0
            total += np.sum(arr[i])
            total += np.sum(arr[:, j])
    #         total += np.heaviside(np.sum(corr[i]), 0)
    #         total += np.heaviside(np.sum(corr[:, j]), 0)
            norm[i][j] = arr[i][j]/total
    return norm
        
# Overall normalization
def overall_norm(arr):
    norm = arr-np.min(arr)
    norm = norm/np.max(arr)
    return norm


# ### Load required data & train 2 models with different parameters
# - Meta data for TCGA samples
# - Case name
# - Tumor labels
# - Train respective models

# In[3]:

para1 = (200, 1000, 500, 6, 20) # parameters for model 1
para2 = (200, 1000, 1000, 6, 20) # parameters for model 1
num_models = 2 # number of different models
aa_list = 'ACDEFGHIKLMNPQRSTVWY'

t = 10000 # Number of random test cases
top = 100 # Number of top test cases to choose from each tumor type

thresh = 3 #threshold for cluster size


# In[4]:

case = 'Tophat_Seq30Min300'
para = para1

num_motifs, batch_size, max_num_kmer, kmer_size, encode_size = para
meta = read_meta('../data/TCGA/tcga_clinical_CC.csv')
data_info, train_data, test_data = load_data(meta, case)
tumors = data_info.tumors.tolist()
num_labels = len(tumors)
labels = setup_tumor_types(meta, tumors)
features = format_balance_labels
encode = encode_aa_seq_binary_ext_dim
count = encode_count_genes
model_path = os.path.join('../work/', case+'-%s-%s-%s-%s-%s'%para)
model1 = GeneSwitchModelFast(num_motifs=num_motifs, num_labels=len(tumors), encode_init=encode_size, 
                                   save_path=model_path)


# In[5]:

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


# ### Testing
# 
# - Define functions
# - Generate random kmers and feed them into NN
# - Sort kmers by output prediction score

# In[6]:

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


# In[7]:

# Predictions
iters = 3000
model1.load(x, CS, ys, max_iterations=iters)
pred1 = model1.predict(x, CS)
model2.load(x, CS, ys, max_iterations=iters)
pred2 = model2.predict(x, CS)


# In[8]:

# Computing tumor-specific prediction score
ctype_predictions = []
pred = [pred1, pred2]

for m in range(num_models):
    ctype_predictions.append([])
    for i in range(t):
        ctype_predictions[m].append([])
        for j in range(num_labels):
            ctype_predictions[m][i].append((sigmoid(pred[m][i][j]), binary_to_kmer(x[i][0])))


# In[9]:

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
        for i2 in a[:, 0]:
            float_a.append(float(i2))
            
        #if (i==4): top = 1000
        else: top = (zscore(float_a) > 1).sum() # uses z-score to compute threshold
        
        high_predictions = a[-top:, :] # selects topkmers for sequence logo
        high_predictions = high_predictions[::-1]
        high_predictions_ctype[m].append(high_predictions)


# ### Clustering on real data
# 
# For each cancer type, we take the top kmers and cluster them in order to determine the binding kmers. 

# In[10]:

# Preparing clustering - binary format
precluster = []
for i in range(num_models):
    precluster.append([])
    for c in range(num_labels):
        precluster[i].append([])
        predarray=high_predictions_ctype[i][c]
        for pred in range(len(predarray)):
            #precluster[i][c].append((predarray[pred][0], kmer_to_binary(predarray[pred][1])))
            precluster[i][c].append(np.asarray(kmer_to_binary(predarray[pred][1])).flatten())


# In[11]:

for i in range(13):
    print(len(high_predictions_ctype[0][i]))
    #print('\n')


# In[13]:

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

clusters = []
n_clusters = 20
for i in range(num_models):
    clusters.append([])
    for j in range(num_labels):
        clusters[i].append([])
        for c_index in range(n_clusters):
            clusters[i][j].append([])
        X = np.asarray(precluster[i][j])
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average") #, affinity=metric
        y = model.fit_predict(X)
        for pre_index in range(len(precluster[i][j])):
            clusters[i][j][y[pre_index]].append(high_predictions_ctype[i][j][pre_index])


# In[14]:

thresh = 0
corr = np.zeros((num_labels, num_labels))
for i1 in range(num_labels): #ctype model 1
    for i2 in range(num_labels): #ctype model 2
        pwm1 = []
        pwm2 = []
        #for c1 in range(len(all_clusters2[0][i1])): #cluster model 1
        for c1 in range(n_clusters):
            if (len(clusters[0][i1][c1]) >= thresh):
                pwm1.append(pwm_calc(clusters[0][i1][c1]).flatten())
        #for c2 in range(len(all_clusters2[1][i2])): #cluster model 2
        for c2 in range(n_clusters):
            if (len(clusters[1][i2][c2]) >= thresh):
                pwm2.append(pwm_calc(clusters[1][i2][c2]).flatten())
        if (len(pwm1) == 0 or len(pwm2) == 0):
            corr[i1][i2] = 0
        else:
            corr_matrix = corr2_coeff(np.asarray(pwm1), np.asarray(pwm2))
            #corr[i1][i2] = compute_avg_max(corr_matrix[0:np.asarray(pwm1).shape[0], -np.asarray(pwm2).shape[0]:])
            corr[i1][i2] = compute_avg_max(corr_matrix)

corr_norm = norm_row_col(corr)
#corr_norm = norm_row_col(corr_norm)
corr_norm = overall_norm(corr_norm)


# In[ ]:

norm_row_col(corr_norm)


# In[20]:

plot_weights(corr_norm, 'Correlation')


# In[18]:

plt.subplot(2, 1, 1)
plt.imshow(corr, cmap=plt.get_cmap('Blues'))
plt.ylabel('Model 1')
plt.xlabel('Model 2')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.imshow(corr_norm, cmap=plt.get_cmap('Blues'))
plt.ylabel('Model 1')
plt.xlabel('Model 2')
plt.colorbar()

#plot_weights(np.flip(corr_norm, axis = 0), '')


# In[328]:

corr_100 = (corr, corr_norm)


# In[342]:

corr_500 = (corr, corr_norm)


# In[351]:

corr_1000 = (corr, corr_norm)


# In[361]:

corr_2000 = (corr, corr_norm)


# In[370]:

corr_3000 = (corr, corr_norm)


# In[371]:

all_corrs = (corr_100, corr_500, corr_1000, corr_2000, corr_3000)


# In[381]:

corr_nums = (100, 500, 1000, 2000, 3000)


# In[383]:

n_subplots = 5
fig = plt.figure(figsize=(30, 5))
for i in range(n_subplots):
    plt.subplot(1, n_subplots, i+1)
    plt.imshow(all_corrs[i][1], cmap=plt.get_cmap('Blues'))
    plt.ylabel('Model 1')
    plt.xlabel('Model 2')
    plt.colorbar()
    plt.title('Iterations: ' + str(corr_nums[i]), fontsize = 18)
plt.subplots_adjust(wspace = 0.4)


# In[380]:

plt.imshow(pwm1)


# #### Visualizing cluster analytics using histograms
# - size of clusters
# - number of clusters

# In[121]:

all_clusters2 = clusters
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


# In[440]:

# # Weighting
# thresh = 0
# corr = np.zeros((num_labels, num_labels))
# for i1 in range(num_labels): #ctype model 1
#     for i2 in range(num_labels): #ctype model 2
#         pwm1 = []
#         pwm2 = []
#         #for c1 in range(len(all_clusters2[0][i1])): #cluster model 1
#         for c1 in range(5):
#             pwm1.append(np.multiply(pwm_calc(all_clusters2[0][i1][c1]), len(all_clusters2[0][i1][c1])).flatten())
#         #for c2 in range(len(all_clusters2[1][i2])): #cluster model 2
#         for c2 in range(5):
#             pwm2.append(np.multiply(pwm_calc(all_clusters2[1][i2][c2]), len(all_clusters2[1][i2][c2])).flatten())
#         if (len(pwm1) == 0 or len(pwm2) == 0):
#             corr[i1][i2] = 0
#         else:
#             corr_matrix = np.corrcoef(pwm1, pwm2)
#             corr[i1][i2] = np.amax(corr_matrix[0:np.asarray(pwm1).shape[0], -np.asarray(pwm2).shape[0]:])


# ### Training Checkpoints

# In[299]:

case = 'Tophat_Seq30Min300'
para = para1

num_motifs, batch_size, max_num_kmer, kmer_size, encode_size = para
meta = read_meta('../data/TCGA/tcga_clinical_CC.csv')
data_info, train_data, test_data = load_data(meta, case)
tumors = data_info.tumors.tolist()
num_labels = len(tumors)
labels = setup_tumor_types(meta, tumors)
features = format_balance_labels
encode = encode_aa_seq_binary_ext_dim
count = encode_count_genes
model_path = os.path.join('../work/', case+'-%s-%s-%s-%s-%s'%para)
model1 = GeneSwitchModelFast(num_motifs=num_motifs, num_labels=len(tumors), encode_init=encode_size, 
                                   save_path=model_path)


# In[306]:

model1.load(x, CS, ys, max_iterations=100)


# In[307]:

pred_100 = model1.predict(x, CS)


# In[ ]:

model1 = GeneSwitchModelFast(num_motifs=num_motifs, num_labels=len(tumors), encode_init=encode_size, 
                                   save_path=model_path)


# In[ ]:

pred1 = model1.predict(x, CS)
model2.load(x, CS, ys)
pred2 = model2.predict(x, CS)

