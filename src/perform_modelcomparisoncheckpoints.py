
# coding: utf-8

# In[4]:

# Import classes and function from DeepBCR
from deep_bcr import *


# In[5]:

from datetime import datetime
save_path = '../work/'

START_TIME = datetime.now()
res = []

n = 200 # Number of patients
RE = False # whether to recovere saved results
SL = False # whether to save log files


# In[6]:

def kmer_to_binary(kmer, aa_list='ACDEFGHIKLMNPQRSTVWY'):
    binary = []
    for i in range(len(kmer)):
        char = kmer[i]
        index = aa_list.find(char)
        array = np.zeros(len(aa_list))
        array[index] = 1
        binary.append(array)
    return binary


# In[7]:

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


# In[8]:

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# In[9]:

def train_model():
    # Getting training data
    xs, cs, ys, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size = k, 
                                   num_pos_kmers=p)
    print('Train data', xs.shape, cs.shape, ys.shape)

    # Creating and training the model
    model = MaxSnippetModel()
    model.train(xs.reshape(n,m,-1), cs, ys, save_log=SL, recover=RE); # get weights??
    return model, ref


# In[10]:

# Generates set of random kmers and calculates the prediction score for each

def test_model(model, ref, t, top):
## model = trained model
## ref = array of positive kmers
## t = number of test cases
## top = number of top test cases to select for figure generation
    predictions = []
    for i in range(t):
    #     if i < 2:
    #         rand_kmer = kmer_to_binary(ref[i])
    #     else:
        # generates random test kmer to be "positive"
        rand_kmer, s = np.asarray(generate_rand_kmer(k))

        # fits kmer to valid input dimensions
        x = np.expand_dims(np.expand_dims(rand_kmer, axis=0), axis = 0)
        x = np.tile(x, (n, m, 1, 1))
        CS = np.ones((n, m))

        #computing prediction value
        pred_value = sigmoid(np.average(model.predict(x.reshape(n,m,-1), CS)[0, :]));

        predictions.append((pred_value, s))
        
    # Adds positive kmers
    for i in range(len(ref)):
        poskmer = ref[i]
        # Encodes kmer string into binary input value for prediction
        poskmer_bin = kmer_to_binary(poskmer)
        x_poskmer = np.expand_dims(np.expand_dims(poskmer_bin, axis=0), axis = 0)
        x_poskmer = np.tile(x_poskmer, (n, m, 1, 1))
        CS_poskmer = np.ones((n, m))

        # Calculates prediction score
        pred_value = sigmoid(np.average(model.predict(x_poskmer.reshape(n,m,-1), 
                                                      CS_poskmer)[0]))
        predictions.append((str(pred_value), poskmer))
        
    # Sorts kmers based on prediction score and selects top kmers
    predarray = np.asarray(predictions)
    ind = np.lexsort((predarray[:,1], predarray[:,0]))    
    a = predarray[ind]
    high_predictions = a[-top:, :] # selects topkmers for sequence logo
    high_predictions = high_predictions[::-1]
    return high_predictions


# ### Calculating the PWM

# In[11]:

# PARAMETERS

n = 500 # Number of patients (to train on)
m = 7 # Max number of kmers per patient
p = 2 # Number of positive kmers
k = 4 # Kmer size
num_test_cases = 1000
num_top = 200 # Number of top test cases to select for seq logo clustering
aa_list='ACDEFGHIKLMNPQRSTVWY'


# In[12]:

trained_model, positive_kmers = train_model()


# In[13]:

prediction = test_model(trained_model, positive_kmers, num_test_cases, num_top)


# In[14]:

# prints top non-weighted predictions
print('Positive kmer(s): ' + ' '.join(positive_kmers))
print('High predictions: ')
print('\n'.join(prediction[:, 1].tolist()))
prediction[:, 0]


# In[15]:

prediction


# In[95]:

weights = np.zeros((len(aa_list), k))

total_sum = 0
for i in range(num_top):
    total_sum += float(prediction[i, 0])

for i in range(len(aa_list)):
    for j in range(k):
        weight = 0
        for p in range(num_top):
            if prediction[p, 1][j] == aa_list[i]:
                weight += float(prediction[p, 0])
        weights[i, j] = int(round((weight * 100 / total_sum), 1) * 10)


# In[96]:

weights


# In[97]:

aminoacids = []
for i in range(20):
    aminoacids.append(str(aa_list[i]))


# In[117]:

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


# In[119]:

plot_weights(weights, 'Total weights')


# In[55]:

f = open('weights.txt','w')
for i in range(len(aa_list)):
    f.write(aa_list[i] + ' | ')
    for j in range(k):
        f.write(str(weights[i, j]) + ' ')
    f.write('\n')
f.close()


# In[56]:

# for i in range(len(aa_list)):
#     print(aa_list[i]+'"'+','+'"', end="")


# #### Clustering

# In[121]:

# Calculates similarity score between 2 equal length kmers by finding # of same characters
def kmer_similarity(kmer1, kmer2):
    score = 0
    for i in range(len(kmer1)):
        if kmer1[i] == kmer2[i]:
            score += 1
    return score


# In[144]:

# Determines clusters based on number of clusters
cluster_pred = prediction

num_clusters = 15
all_clusters = []
for i in range(num_clusters):
    cluster = []
    top_kmer = cluster_pred[0, 1]
    for j in range(len(cluster_pred)):
        if j >= len(cluster_pred): break
        if kmer_similarity(top_kmer, cluster_pred[j, 1]) > 1: #can change similarity calc
            cluster.append(cluster_pred[j])
            cluster_pred = np.delete(cluster_pred, j, 0)
    all_clusters.append(cluster)


# In[145]:

clustered_weights = np.zeros((num_clusters, len(aa_list), k))
clusters = np.asarray(all_clusters)

for c in range(len(all_clusters)):
    total_sum = 0
    for i in range(len(all_clusters[c])):
        total_sum += float(clusters[c][i][0])

    for i in range(len(aa_list)):
        for j in range(k):
            weight = 0
            for p in range(len(all_clusters[c])):
                if all_clusters[c][p][1][j] == aa_list[i]:
                    weight += float(all_clusters[c][p][0])
            clustered_weights[c, i, j] = int(round((weight * 100 / total_sum), 1) * 10)


# In[146]:

# def plot_cluster_weights(weights, cluster):
#     plot_weights(weights)
#     plt.title('Cluster ' + str(i+1))
#     plt.show()


# In[150]:

# for i in range(num_clusters):
#     plt.subplot(1, num_clusters, i+1)
#     title = 'Cluster ' + str(i+1)
#     plot_weights(clustered_weights[i], title)
# plt.show()


# In[148]:

# Visualize weights
for i in range(num_clusters):
    plt.subplot(1, num_clusters, i+1)
    plt.imshow(clustered_weights[i], cmap = 'hot')
    #fig, ax = plt.subplots(figsize=(10, 10))
    plt.title('Cluster ' + str(i+1))
plt.subplots_adjust(wspace = 1)
plt.show()


# In[149]:

# protein format
for c in range(num_clusters):
    filename = 'cynthia' + str(c+1) + '.txt'
    f = open(filename, 'w')
    f.write('\t')
    for i in range(20):
        f.write(aa_list[i] + ' ')
    f.write('\n')
    for i in range(k):
        f.write(str(i+1) + ' \t')
        for j in range(len(aa_list)):
            f.write(str(clustered_weights[c][j][i]) + ' ')
        f.write('\n')
    f.close()


# In[ ]:

# MEME format
# for c in range(num_clusters):
#     filename = 'cynthia' + str(c+1) + '.meme'
#     f = open(filename, 'w')
#     f.write('MEME version 4.5\n')
#     f.write('ALPHABET= ACDEFGHIKLMNPQRSTVWY\n')
#     f.write('END ALPHABET\n')
#     f.write('Background letter frequencies (from unknown):\n')
#     for i in range(len(aa_list)):
#         f.write(aa_list[i] + ' 0.05 ')
#     f.write('\nMOTIF testmotif_' + str(c) + '\n')
#     f.write('letter-probability matrix: alength= 20 w= 4 nsites= 18 E= 0\n')
#     for i in range(4):
#         for j in range(20):
#             f.write('0.05 ')
#         f.write('\n')
#     for i in range(len(aa_list)):
#         f.write(aa_list[i] + ' | ')
#         for j in range(k):
#             f.write(str(weights[c][i][j]) + ' ')
#         f.write('\n')
#     f.close()


# In[2]:

from plotRadialProtein.r import *


# In[ ]:
