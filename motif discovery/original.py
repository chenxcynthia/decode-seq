
# coding: utf-8

# # Deep learning models for BCR repretoires in TCGA samples

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

from tcga_bcr import *
from deep_bcr import *
from tcga_bcr_comp import plot_figures, plot_roc


# ### Load required data
# - Meta data for TCGA samples
# - Case name
# - Train and test datasets
# - Model and model parameters

# In[2]:

case = 'Tophat_Seq30Min300'
para = (200, 1000, 500, 6, 20)

num_motifs, batch_size, max_num_kmer, kmer_size, encode_size = para


# In[3]:

meta = read_meta('../data/TCGA/tcga_clinical_CC.csv')


# In[4]:

data_info, train_data, test_data = load_data(meta, case)


# In[5]:

tumors = data_info.tumors.tolist()
data_info


# ### Check the training data
# Show the number of k-mers in samples of patients with different cancer types. This may help us to set a reasonable parameter for the number of k-mers to sample in a repertoire.

# In[6]:

labels = setup_tumor_types(meta, tumors)

# ## para size=None means to get all samples
# dx, dc, dy = next(get_next_batch(train_data, labels, size=None, kmer=kmer_size))
# lens = [len(i) for i in dx]
# df = pd.DataFrame(np.array([lens, dy]).T, columns=['PoolSize', 'Label'])

# axe = df.boxplot('PoolSize', by='Label', grid=False)
# axe.set_ylim([0, 10000])
# axe.set_xticklabels(tumors);
# plt.setp(plt.xticks()[1], rotation=30, ha='right')
# plt.xlabel('')
# plt.ylabel('Number of kmers')
# plt.xlabel('Cancer type')
# plt.title('')
# plt.show()


# ### Example of train and test
# 1. Train a model from a random initalization

# In[7]:

labels = setup_tumor_types(meta, tumors)
features = format_balance_labels
encode = encode_aa_seq_binary_ext_dim
count = encode_count_genes

model_path = os.path.join('../work/', case+'-%s-%s-%s-%s-%s'%para)
model = GeneSwitchModelFast(num_motifs=num_motifs, num_labels=len(tumors), 
                            encode_init=encode_size, save_path=model_path)

#train_iter = sample_data_iter(train_data, batch_size, kmer_size, max_num_kmer, labels, features, encode, count)

#for i in range(30):
#    train = model.train_batch(train_iter, max_iterations=(i+1)*100, it_step_size=20)


# 2. Test the model using the `test()` function

# In[8]:

data = next(get_next_batch(test_data, labels, size=None, kmer=kmer_size))
for i in range(1):
    xs, cs, ys = format_trim_dims(data, dim=max_num_kmer, encode=encode, count=count)
#    ys_pred, ys_obj, ys_acc = model.test(xs, cs, ys)
#    print(ys_pred.shape)


# 3. Test the model using the `load()` and `predict()` functions

# In[9]:

loaded_model = GeneSwitchModelFast(num_motifs=num_motifs, num_labels=len(tumors), encode_init=encode_size, 
                                   save_path=model_path)

data = next(get_next_batch(test_data, labels, size=None, kmer=kmer_size))
for i in range(1):
    xs, cs, ys = format_trim_dims(data, dim=max_num_kmer, encode=encode, count=count)
    loaded_model.load(xs, cs, ys)
    ys_pred = loaded_model.predict(xs, cs)
    print(ys_pred.shape)


# ### Create sequence logos from real data
# 
# - Define functions
# - Generate random kmers and feed them into NN
# - Sort kmers by output prediction score

# In[10]:

import numpy as np
from deep_bcr import *
from seq_logo_functions import *


# In[11]:

# Functions for processing and preparing kmer data

def binary_to_kmer(binary, aa_list='ACDEFGHIKLMNPQRSTVWY'):
    kmer = ""
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


# In[12]:

# Predictions all at once

t = 10000
top = 20
num_labels = len(tumors)

x = []
for i in range(t):
    # generates random test kmer to be "positive"
    rand_kmer, s = np.asarray(generate_rand_kmer(kmer_size))
    x.append(rand_kmer)

# fits xs to valid input dimensions
x = np.expand_dims(np.asarray(x), axis = 1)
CS = np.ones((t, 1, 5))
ys = np.ones((t, num_labels))

loaded_model.load(x, CS, ys)
pred = loaded_model.predict(x, CS)#[0, :] # prediction array


# In[13]:

# Sorting predictions by cancer type

ctype_predictions = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range(t):
    #pred_value = sigmoid(np.amax(pred[i]))
    #cancer_type = np.argmax(pred[i]) # based on highest scoring label
    #ctype_predictions[cancer_type].append((pred_value, binary_to_kmer(x[i][0])))
    for j in range(num_labels):
        ctype_predictions[j].append((sigmoid(pred[i][j]), binary_to_kmer(x[i][0])))
    
# Sorting each cancer type by prediction score
high_predictions_ctype = []
represented_ctypes = []
for i in range(num_labels):
    # Sorts kmers based on prediction score and selects top kmers
    predarray = np.asarray(ctype_predictions[i])
    if len(predarray) == 0:
        continue
    represented_ctypes.append(i)
    ind = np.lexsort((predarray[:,1], predarray[:,0]))    
    a = predarray[ind]
    high_predictions = a[-top:, :] # selects topkmers for sequence logo
    high_predictions = high_predictions[::-1]
    high_predictions_ctype.append(high_predictions)


# In[14]:

for i in range(len(represented_ctypes)):
    print('Cancer type: ' + tumors[i])
    print('\n'.join(high_predictions_ctype[i][:, 1].tolist()) + '\n')


# ### Clustering on real data
# 
# For each cancer type, we take the top kmers and cluster them in order to determine the binding kmers. 

# In[15]:

def kmer_similarity(kmer1, kmer2):
    score = 0
    for i in range(len(kmer1)):
        if kmer1[i] == kmer2[i]:
            score += 1
    return score


# In[17]:

# New sorting + selecting method which takes only unique kmers
high_predictions_ctype_cluster = []
top = 100
for i in range(num_labels):
    predarray = np.asarray(ctype_predictions[i])
    #predarray = np.unique(predarray, axis = 0)
    ind = np.lexsort((predarray[:,1], predarray[:,0]))    
    a = predarray[ind]
    high_predictions = a[-top:, :] # selects topkmers for sequence logo
    high_predictions = high_predictions[::-1]
    high_predictions_ctype_cluster.append(high_predictions)


# In[18]:

# Different clustering method
# Forms clusters until there are no similar kmers

all_clusters2 = []
cluster_motifs = []
top_motifs = 5

for c in range(num_labels):
    cluster_pred = high_predictions_ctype_cluster[c]
    clusters = []
    
    motifs = []

    while (len(cluster_pred)>0):
        cluster = []
        top_kmer = cluster_pred[0, 1]
        motif_sum = 0
        j = 0
        while j < len(cluster_pred):
            if kmer_similarity(top_kmer, cluster_pred[j, 1]) > 2:
                cluster.append(cluster_pred[j, 1])
                motif_sum += float(cluster_pred[j, 0])
                cluster_pred = np.delete(cluster_pred, j, 0)
                j = j-1
            j += 1
        if len(cluster) > 1: # Only clusters with 2+ size
            clusters.append(cluster)
            motifs.append((top_kmer, motif_sum))
            
    all_clusters2.append(clusters)
    cluster_motifs.append(motifs[0:top_motifs])


# In[ ]:

#all_clusters2


# In[19]:

## Prints clusters
for c in range(num_labels):
    print('Cancer type: ' + tumors[c])
    for i in range(len(all_clusters2[c])):
        print('Cluster ' + str(i+1) + ':')
        print('\n'.join(all_clusters2[c][i]))
    print('\n')


# In[20]:

# Create seperate PWM files for the cancer types

for c in range(num_labels):
    filename = 'pwm_top' + str(c+1) + '.txt'
    f = open(filename, 'w')
    f.write('\t')
    for i in range(20):
        f.write(aa_list[i] + ' ')
    f.write('\n')
    for i in range(cdr3_length):
        f.write(str(i+1) + ' \t')
        for j in range(len(aa_list)):
            f.write(str(pwm_top[c][j][i]) + ' ')
        f.write('\n')
    f.close()


# In[21]:

# Write cluster results to a file
filename = 'clusters_' + str(kmer_size) + 'mer_2.txt'
f = open(filename, 'w')

for c in range(num_labels):
    f.write('Cancer type: ' + tumors[c])
    f.write('\n')
    for i in range(len(all_clusters2[c])):
        if(len(all_clusters2[c][i]) <= 2): continue
        f.write('Cluster ' + str(i+1) + ': ')
        f.write(' '.join(all_clusters2[c][i]))
        f.write('\n')
    f.write('\n')
    f.write('\n')
f.close()


# In[22]:

np.asarray(high_predictions_ctype_cluster)[:, :, 1].flatten()


# In[23]:

high_predictions_ctype_cluster[0][:, 1]


# #### Determining positive motifs for each cancer type

# In[24]:

import pandas as pd
# Goal: to print a table where 


# In[25]:

motifs = np.asarray(cluster_motifs)[:,:,0]
motif_scores = np.asarray(cluster_motifs)[:,:,1]
motif_df = pd.DataFrame(motifs)


# In[28]:

for i in range(len(motif_scores)):
    scoresum = 0
    for score in motif_scores[i]:
        scoresum += float(score)
    #total = np.sum(motif_scores[i])
    for j in range(len(motif_scores[0])):
        motif_scores[i][j] = float(motif_scores[i][j]) / scoresum
motif_scores = motif_scores.astype(np.float)


# In[32]:

import matplotlib.pyplot as plt
import numpy as np
randn = np.random.randn
from pandas import *
import pylab as pl


# In[35]:

idx = Index(np.arange(1,num_labels+1))
df = motif_df
vals = np.around(motif_scores, 2)
normal = plt.Normalize(vals.min()-0.05, vals.max())
colLabels = []
for i in range(top_motifs):
    colLabels.append('Cluster ' + str(i+1))


# In[36]:

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# your dataset
nValues = np.arange(0,30)
xValues = np.linspace(0,10)
dataset = [(xValues-5-0.5*n)**2 for n in nValues]

# setup the normalization and the colormap
normalize = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
colormap = cm.bwr

# plot
# for n in nValues:
#     plt.plot(dataset[n], color=colormap(normalize(n)))
    
the_table=plt.table(cellText=motifs, rowLabels=tumors, colLabels=colLabels, 
                    colWidths = [0.18]*vals.shape[1], loc='center', cellLoc = 'center',
                    cellColours=plt.cm.bwr(normal(vals)))

# setup the colorbar
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(vals)
plt.colorbar(scalarmappaple, shrink=0.7)

# cbar = ax.figure.colorbar(scalarmappaple)
# cbar.ax.set_ylabel("Motif scores", rotation=-90, va="bottom")

plt.axis('off')
plt.savefig("figures/motiftable.png", bbox_inches='tight', dpi = 500)
# show the figure
# plt.show()


# In[20]:

# idx = Index(np.arange(1,num_labels+1))
# df = motif_df
# vals = np.around(motif_scores, 2)
# normal = plt.Normalize(vals.min()-0.05, vals.max())

# fig = plt.figure(figsize=(11, 4))
# ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
# #ax.set_xlabel('Clusters')
# #ax.set_ylabel('Tumor type')
# colLabels = []
# for i in range(top_motifs):
#     colLabels.append('Cluster ' + str(i+1))

# the_table=plt.table(cellText=motifs, rowLabels=tumors, colLabels=colLabels, 
#                     colWidths = [0.18]*vals.shape[1], loc='center', cellLoc = 'center',
#                     cellColours=plt.cm.Reds(normal(vals)))
# #plt.colorbar(vals)

# import pylab as pl
# #pl.figure(figsize=(11, 4))
# #pl.figure(figsize=(7, 7))
# #pl.annotate(..., fontsize=1, ...)
# pl.savefig('test.pdf', format='pdf')


# In[392]:

#plt.figure(figsize = (20, 10))
plt.imshow(vals, cmap=plt.get_cmap('Reds'))
plt.colorbar()


# In[346]:

def plot_weights(weights, plot_title):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(weights, cmap='hot')

    # Set tick labels
    residues = list(range(4))
    residues = [1, 2, 3, 4]
    ax.set_xticks(np.arange(len(residues)))
    ax.set_yticks(np.arange(len(aminoacids)))
    ax.set_xticklabels(residues)
    ax.set_yticklabels(aminoacids);

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
    plt.title(plot_title, fontsize=20)
    
    from matplotlib import rcParams
    rcParams['axes.titlepad'] = 20 

    plt.show()


# In[347]:

plot_weights(motif_scores, 'Total weights')


# In[ ]:

for y in range(data.shape[0]):
    for x in range(data.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % data[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

