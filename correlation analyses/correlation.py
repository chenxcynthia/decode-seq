
# coding: utf-8

# # Single script for all model comparison

# ### Load required packages and functions
# - Tensorflow package
# - Sklearn package
# - Functions in tcga_bcr.py
# 

# In[2]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from scipy.stats.mstats import zscore
from matplotlib.pyplot import colorbar
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from mpl_toolkits.mplot3d import Axes3D

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

# In[3]:

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
    
# Authors: Mathew Kallada
# License: BSD 3 clause
"""
=========================================
Plot Hierarachical Clustering Dendrogram 
=========================================
This example plots the corresponding dendrogram of a hierarchical clustering
using AgglomerativeClustering and the dendrogram method available in scipy.
"""

from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
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

# Computes a performance reliability score by taking difference between diagonal + other values
def compute_matrix_score(matrix):
    # square matrix
    diag = 0
    others = 0
    length = len(matrix)
    for i in range(length):
        for j in range(length): 
            if i==j:
                diag += matrix[i][j]
            else:
                others += matrix[i][j]
                
    diag = diag/length
    others = others/(length*length-length)
    return diag-others


# ### Load required data & train 2 models with different parameters
# - Meta data for TCGA samples
# - Case name
# - Tumor labels
# - Train respective models

# In[4]:

para1 = (200, 1000, 500, 6, 20) # parameters for model 1
para2 = (200, 1000, 1000, 6, 20) # parameters for model 1
num_models = 2 # number of different models
aa_list = 'ACDEFGHIKLMNPQRSTVWY'


# In[5]:

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


# In[6]:

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


# ### Testing with Training Checkpoints
# 
# - Define functions
# - Generate random kmers and feed them into NN
# - Sort kmers by output prediction score

# In[7]:

# Testing parameters
t = 20000 # Number of random test cases
top = 100 # Number of top test cases to choose from each tumor type
#thresh = 3 #threshold for cluster size
n_clusters = 10


# In[8]:

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


# In[19]:

200*np.asarray(range(10))+200


# In[20]:

iters = 200*np.asarray(range(10))+200
num_iters = len(iters)
all_corrs = []
all_corrs2 = []
all_pwms = [[], []]


# In[38]:

len(predarray)


# In[21]:

for iter_index in range(num_iters):
    model1.load(x, CS, ys, max_iterations=iters[iter_index])
    pred1 = model1.predict(x, CS)
    model2.load(x, CS, ys, max_iterations=iters[iter_index])
    pred2 = model2.predict(x, CS)

    # Computing tumor-specific prediction score
    ctype_predictions = []
    pred = [pred1, pred2]
    for m in range(num_models):
        ctype_predictions.append([])
        for i in range(t):
            ctype_predictions[m].append([])
            for j in range(num_labels):
                ctype_predictions[m][i].append((sigmoid(pred[m][i][j]), binary_to_kmer(x[i][0])))

    # Sorting each cancer type by prediction score
    high_predictions_ctype = []
    for m in range(num_models):
        high_predictions_ctype.append([])
        for i in range(num_labels):
            # Sorts kmers based on prediction score and selects top kmers
            predarray = np.asarray(ctype_predictions)[m,:,i]
#             if len(predarray) == 0:
#                 continue
            ind = np.lexsort((predarray[:,1], predarray[:,0]))    
            a = predarray[ind]

            float_a = []
            for i2 in a[:, 0]:
                float_a.append(float(i2))
            top = (zscore(float_a) > 1).sum() # uses z-score to compute threshold, and then finds top indices

            high_predictions = a[-top:, :] # selects topkmers for sequence logo
            high_predictions = high_predictions[::-1]
            high_predictions_ctype[m].append(high_predictions)

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

    clusters = []
    for i in range(num_models):
        clusters.append([])
        for j in range(num_labels):
            clusters[i].append([])
            for c_index in range(n_clusters):
                clusters[i][j].append([])
            X = np.asarray(precluster[i][j])
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
            y = model.fit_predict(X)
            for pre_index in range(len(precluster[i][j])):
                clusters[i][j][y[pre_index]].append(high_predictions_ctype[i][j][pre_index])

    thresh = 0
    corr = np.zeros((num_labels, num_labels))
    for i1 in range(num_labels): #ctype model 1
        all_pwms[0].append([])
        all_pwms[1].append([])
        for i2 in range(num_labels): #ctype model 2
            all_pwms[0][i1].append([])
            all_pwms[1][i1].append([])
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
            all_pwms[0][i1][i2].append(pwm1)
            all_pwms[1][i1][i2].append(pwm2)
            
            
            
            if (len(pwm1) == 0 or len(pwm2) == 0):
                corr[i1][i2] = 0
            else:
                corr_matrix = corr2_coeff(np.asarray(pwm1), np.asarray(pwm2))
                corr[i1][i2] = compute_avg_max(corr_matrix)

    corr_norm = norm_row_col(corr)
    #corr_norm = norm_row_col(corr_norm)
    corr_norm = overall_norm(corr_norm)
    all_corrs2[iter_index1].append((corr, corr_norm))


# In[25]:

thresh = 0
corr = np.zeros((num_labels, num_labels))
for i1 in range(num_labels): #ctype model 1
    all_pwms[0].append([])
    all_pwms[1].append([])
    for i2 in range(num_labels): #ctype model 2
        all_pwms[0][i1].append([])
        all_pwms[1][i1].append([])
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
        all_pwms[0][i1][i2].append(pwm1)
        all_pwms[1][i1][i2].append(pwm2)
        
        
        
        if (len(pwm1) == 0 or len(pwm2) == 0):
            corr[i1][i2] = 0
        else:
            corr_matrix = corr2_coeff(np.asarray(pwm1), np.asarray(pwm2))
            corr[i1][i2] = compute_avg_max(corr_matrix)

corr_norm = norm_row_col(corr)
#corr_norm = norm_row_col(corr_norm)
corr_norm = overall_norm(corr_norm)
all_corrs2[iter_index1].append((corr, corr_norm))


# In[23]:

len(all_pwms[0])


# In[110]:

len(precluster[0][0])


# In[111]:

X = np.asarray(precluster[0][0][0:100])
model = AgglomerativeClustering(n_clusters=10, linkage="average")
model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(model, labels=model.labels_)
plt.show()


# In[79]:

np.asarray(all_pwms[0]).shape


# In[49]:

for iter_index1 in range(num_iters):
    all_corrs2.append([])
    model1.load(x, CS, ys, max_iterations=iters[iter_index1])
    pred1 = model1.predict(x, CS)
    for iter_index2 in range(num_iters):
        model2.load(x, CS, ys, max_iterations=iters[iter_index2])
        pred2 = model2.predict(x, CS)

        # Computing tumor-specific prediction score
        ctype_predictions = []
        pred = [pred1, pred2]

        for m in range(num_models):
            ctype_predictions.append([])
            for i in range(t):
                ctype_predictions[m].append([])
                for j in range(num_labels):
                    ctype_predictions[m][i].append((sigmoid(pred[m][i][j]), binary_to_kmer(x[i][0])))

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

        clusters = []
        for i in range(num_models):
            clusters.append([])
            for j in range(num_labels):
                clusters[i].append([])
                for c_index in range(n_clusters):
                    clusters[i][j].append([])
                X = np.asarray(precluster[i][j])
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
                y = model.fit_predict(X)
                for pre_index in range(len(precluster[i][j])):
                    clusters[i][j][y[pre_index]].append(high_predictions_ctype[i][j][pre_index])

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
                    corr[i1][i2] = compute_avg_max(corr_matrix)

        corr_norm = norm_row_col(corr)
        #corr_norm = norm_row_col(corr_norm)
        corr_norm = overall_norm(corr_norm)
        all_corrs2[iter_index1].append((corr, corr_norm))


# In[52]:

model_corrs = np.zeros((num_iters, num_iters))
for i in range(num_iters):
    for j in range(num_iters):
        model_corrs[i][j] = compute_matrix_score(all_corrs2[i][j][1])


# In[53]:

model_corrs


# In[85]:

X = []
Y = []
for i in range(num_iters):
    X.append(np.arange(num_iters))
    y_row = []
    for j in range(num_iters):
        y_row.append(i)
    Y.append(y_row)
Z = model_corrs


# In[88]:

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, Z, cmap=plt.cm.Reds)
ax.plot_wireframe(X, Y, Z, color='white', linewidths=0.5)
ax.view_init(20, 310)
ax.set_xlabel('Model 1')
ax.set_ylabel('Model 2')
ax.set_zlabel('Correlation')
ax.xaxis._axinfo['label']['space_factor'] = 2.8
stops = np.arange(5)
plt.show()


# In[93]:

plt.imshow(Z, cmap=plt.get_cmap('Reds'))
plt.colorbar()


# In[96]:

len(high_predictions_ctype[0])


# In[118]:

# Calculate heatmap for overlapping kmers
overlapping = np.zeros((num_labels, num_labels))
for i in range(num_labels):
    for j in range(num_labels):
        a = high_predictions_ctype[0][i][:, 1]
        b = high_predictions_ctype[1][j][:, 1]
        overlapping[i][j] = len(list(set(a) & set(b))) / (len(a)+len(b))


# In[ ]:

plt.imshow(norm_row_col(overlapping), cmap=plt.get_cmap('Reds'))
plt.colorbar()


# In[119]:

plt.imshow(overlapping, cmap=plt.get_cmap('Reds'))
plt.colorbar()


# In[13]:

save = all_corrs


# In[14]:

n_subplots = len(iters)
fig = plt.figure(figsize=(30, 5))
for i in range(n_subplots):
    plt.subplot(1, n_subplots, i+1)
    plt.imshow(all_corrs[i][1], cmap=plt.get_cmap('Blues'))
    plt.ylabel('Model 1')
    plt.xlabel('Model 2')
    plt.colorbar()
    plt.title('Iterations: ' + str(iters[i]), fontsize = 18)
plt.subplots_adjust(wspace = 0.4)


# In[17]:

matrix_scores = []
for i in range(n_subplots):
    matrix_scores.append(compute_matrix_score(save[i][1]))


# In[18]:

plt.plot(matrix_scores)


# In[19]:

matrix_scores

