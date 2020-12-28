
# coding: utf-8

# In[12]:

# Import classes and function from DeepBCR
from deep_bcr import *


# In[13]:

from datetime import datetime
save_path = '../work/'

START_TIME = datetime.now()
res = []

n = 200 # Number of patients
RE = False # whether to recovere saved results
SL = False # whether to save log files

# for m in [10, 50, 100]: # Max number of kmers per patient 
#     for i in [1,2,4,8,16]: # Number of positive kmers
#         model_path = os.path.join(save_path, 'num%s_max%s_pos%s/'%(n,m,i))
#         if not os.path.exists(model_path):
#             os.mkdir(model_path)
#         print(m, '--------------', i)
#         xs, cs, ys, ref = get_syn_data(num_samples=n, num_kmers=m, num_pos_kmers=i)
#         print('Train data', xs.shape, cs.shape, ys.shape)
#         XS, CS, YS, ref = get_syn_data(num_samples=n, num_kmers=m, num_pos_kmers=i,
#                                        positive=ref)
#         print('Test data', XS.shape, CS.shape, YS.shape)

#         break  ## no need for a full test
#     break

# res = pd.DataFrame(res, columns=['#Samples','#Snips','#PosCases','Model',
#                                  'Test_obj','Test_acc'])
# res.to_csv(os.path.join(save_path, 'result_compare.csv'), index=False)
# print('Models are saved in', save_path)

# FINISH_TIME = datetime.now()
# print('Start  at', START_TIME)
# print('Finish at', FINISH_TIME)
# print("Time Cost", FINISH_TIME-START_TIME)


# In[14]:

def kmer_to_binary(kmer, aa_list='ACDEFGHIKLMNPQRSTVWY'):
    binary = []
    for i in range(len(kmer)):
        char = kmer[i]
        index = aa_list.find(char)
        array = np.zeros(len(aa_list))
        array[index] = 1
        binary.append(array)
    return binary


# In[15]:

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


# In[16]:

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# In[23]:

# PARAMETERS

n = 200 # Number of patients (to train on)
m = 7 # Max number of kmers per patient
p = 1 # Number of positive kmers
k = 6 # Kmer size
aa_list='ACDEFGHIKLMNPQRSTVWY' # List of amino acids for 


# In[28]:

#def train_model():
# Getting training data
xs, cs, ys, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size = k, num_pos_kmers=p)
print('Train data', xs.shape, cs.shape, ys.shape)

# Getting testing data based on positive kmer
XS, CS, YS, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size = k, num_pos_kmers=p,
                                           positive=ref)
print('Test data', XS.shape, CS.shape, YS.shape)

# Creating and training the model
model = MaxSnippetModel()
model.train(xs.reshape(n,m,-1), cs, ys, save_log=SL, recover=RE); # get weights??
#return model


# In[30]:

# Loading the model

new_n = 100
new_m = 7

x = np.zeros((n, m, k * len(aa_list)))
c = np.zeros((n, m))
y = np.zeros(n)
model.load(x, c, y)


# In[31]:

# train_model()


# In[32]:

# Inputting the positive kmer(s) to validate neural net results
   
# Tests every positive kmer in ref array
for i in range(len(ref)):
    poskmer = ref[i]
    # Encodes kmer string into binary input value for prediction
    poskmer_bin = kmer_to_binary(poskmer)
    poskmer_bin = np.expand_dims(np.expand_dims(poskmer_bin, axis=0), axis = 0)
    poskmer_bin = np.tile(poskmer_bin, (n, m, 1, 1))

    # Calculates prediction score
    pred_value = model.predict(XS.reshape(n,m,-1), CS)

    #Calculates 0-1 confidence level from logits
    #m1 model
    r = len(pred_value)
    c = len(pred_value[0])
    sig_pred = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            sig_pred[i][j] = sigmoid(pred_value[i][j])

#     # m2 model
#     for i in range(r):
#         sig_pred[i] = sigmoid(pred_value[i])

    # Prints prediction score
    print(sig_pred)


# In[438]:

# Generates set of random kmers and calculates the prediction score for each

num_test_cases = 100
predictions = []

for i in range(num_test_cases):
    # generates random test kmer to be "positive"
    rand_kmer, s = np.asarray(generate_rand_kmer(k))
    
    # fits kmer to valid input dimensions
    x = np.expand_dims(np.expand_dims(rand_kmer, axis=0), axis = 0)
    x = np.tile(x, (n, m, 1, 1))
    
    #computing prediction value
    pred_value = sigmoid(np.average(model.predict(x.reshape(n,m,-1), CS)[0, :]));
    
    predictions.append((pred_value, s))


# In[439]:

# Sorts kmers based on prediction score and selects top kmers
num_top = 10 # number of top kmers to select

predarray = np.asarray(predictions)
ind = np.lexsort((predarray[:,1], predarray[:,0]))    
a = predarray[ind]
high_predictions = a[-num_top:, :] # selects topkmers for sequence logo

# prints top non-weighted predictions
print('Positive kmer(s): ' + ' '.join(ref))
print('Unweighted predictions: ')
print('\n'.join(high_predictions[:, 1].tolist()))
high_predictions[:, 0]


# In[440]:

# Adding weights to the kmers based on magnitude of prediction score
# Assigns weights by increasing frequency based on score
# Ex: score of 0.9 -> 9 copies of that kmer

weighted_pred = []
for i in range(num_top):
    # Calculate "weight"
    freq = (int(round(float(high_predictions[i, 0]), 1) * 10))
    print(freq)
    # Add 'freq' number of kmers
    for j in range(freq):
        weighted_pred.append(high_predictions[i, 1])
        
# prints top weighted predictions
print('Positive kmer(s): ' + poskmer)
print('Weighted predictions: ')
print('\n'.join(weighted_pred))


# ### Multi-model

# In[442]:

# PARAMETERS

n = 250 # Number of patients (to train on)
m = 7 # Max number of kmers per patient
p = 2 # Number of positive kmers
k = 6 # Kmer size
num_test_cases = 500
num_top = 10 # Number of top test cases to select for seq logo


# In[11]:

# Getting training data
xs, cs, ys, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size = k, num_pos_kmers=p)
print('Train data', xs.shape, cs.shape, ys.shape)

# Getting testing data based on positive kmer
XS, CS, YS, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size = k, num_pos_kmers=p,
                                           positive=ref)
print('Test data', XS.shape, CS.shape, YS.shape)

# Creating, training, and testing the model
models = (MaxSnippetModel(), TwoLayerModel())

for i in range(len(models)):
    models[i].train(xs.reshape(n,m,-1), cs, ys, save_log=SL, recover=RE) # get weights??
    pred_value = models[i].predict(XS.reshape(n,m,-1), CS)

    predictions = []
    for j in range(num_test_cases):
        # generates random kmer to test
        rand_kmer, s = np.asarray(generate_rand_kmer(k))
        # fits kmer to valid input dimensions
        x = np.expand_dims(np.expand_dims(rand_kmer, axis=0), axis = 0)
        x = np.tile(x, (n, m, 1, 1))
        # computing prediction value and adding it to prediction array
        pred_value = sigmoid(np.mean(models[i].predict(x.reshape(n,m,-1), CS).flatten()));
        predictions.append((pred_value, s))
        
    predarray = np.asarray(predictions)
    ind = np.lexsort((predarray[:,1], predarray[:,0]))    
    a = predarray[ind]
    high_predictions = a[-num_top:, :] # selects topkmers for sequence logo

    # prints top non-weighted predictions
    print('Positive kmer(s): ' + ' '.join(ref))
    print('Unweighted predictions: ')
    print('\n'.join(high_predictions[:, 1].tolist()))
    high_predictions[:, 0]
    
    weighted_pred = []
    for i in range(num_top):
        # Calculate "weight"
        freq = (int(round(float(high_predictions[i, 0]), 1) * 10))
        print(freq)
        # Add 'freq' number of kmers
        for j in range(freq):
            weighted_pred.append(high_predictions[i, 1])

    # prints top weighted predictions
    print('Weighted predictions: ')
    print('\n'.join(weighted_pred))
    


# ### Clustering

# In[2]:

# Calculates similarity score between 2 equal length kmers by finding # of same characters
def kmer_similarity(kmer1, kmer2):
    score = 0
    for i in range(len(kmer1)):
        if kmer1[i] == kmer2[i]:
            score += 1
    return score


# In[18]:

# PARAMETERS

n = 250 # Number of patients (to train on)
m = 7 # Max number of kmers per patient
p = 2 # Number of positive kmers
k = 4 # Kmer size
num_test_cases = 200
num_top = 10 # Number of top test cases to select for seq logo clustering


# In[20]:

# Getting training data
xs, cs, ys, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size = k, num_pos_kmers=p)
print('Train data', xs.shape, cs.shape, ys.shape)

# Getting testing data based on positive kmer
XS, CS, YS, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size = k, num_pos_kmers=p,
                                           positive=ref)
print('Test data', XS.shape, CS.shape, YS.shape)

# Creating and training the model
model = MaxSnippetModel()
model.train(xs.reshape(n,m,-1), cs, ys, save_log=SL, recover=RE);


# In[100]:

# Generates set of random kmers and calculates the prediction score for each

predictions = []
for i in range(num_test_cases):
#     if i < 2:
#         rand_kmer = kmer_to_binary(ref[i])
#     else:
    # generates random test kmer to be "positive"
    rand_kmer, s = np.asarray(generate_rand_kmer(k))
    
    # fits kmer to valid input dimensions
    x = np.expand_dims(np.expand_dims(rand_kmer, axis=0), axis = 0)
    x = np.tile(x, (n, m, 1, 1))
    
    #computing prediction value
    pred_value = sigmoid(np.average(model.predict(x.reshape(n,m,-1), CS)[0, :]));
    
    predictions.append((pred_value, s))


# In[101]:

# Adds positive kmers
for i in range(len(ref)):
    poskmer = ref[i]
    # Encodes kmer string into binary input value for prediction
    poskmer_bin = kmer_to_binary(poskmer)
    poskmer_bin = np.expand_dims(np.expand_dims(poskmer_bin, axis=0), axis = 0)
    poskmer_bin = np.tile(poskmer_bin, (n, m, 1, 1))

    # Calculates prediction score
    pred_value = sigmoid(np.average(model.predict(XS.reshape(n,m,-1), CS)[0]))
    predictions.append((str(pred_value), poskmer))


# In[121]:

# Sorts kmers based on prediction score and selects top kmers
predarray = np.asarray(predictions)
ind = np.lexsort((predarray[:,1], predarray[:,0]))    
a = predarray[ind]
high_predictions = a[-num_top:, :] # selects topkmers for sequence logo

high_predictions = high_predictions[::-1]
high_predictions = a[-num_top:, :]
high_predictions = high_predictions[::-1]


# In[103]:

from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# In[117]:

# Determines clusters based on number of clusters

num_clusters = 2
all_clusters = []
for i in range(num_clusters):
    cluster = []
    top_kmer = high_predictions[0, 1]
    for j in range(len(high_predictions)):
        if j >= len(high_predictions): break
        if kmer_similarity(top_kmer, high_predictions[j, 1]) > 1: #can change similarity calc
            cluster.append(high_predictions[j, 1])
            high_predictions = np.delete(high_predictions, j, 0)
    all_clusters.append(cluster)


# In[118]:

all_clusters


# In[122]:

# Different clustering method
# Forms lusters until there are no similar kmers

cluster_pred = high_predictions

all_clusters2 = []
#b = True
while (b and (len(cluster_pred)>0)):
    #b = False
    cluster = []
    top_kmer = cluster_pred[0, 1]
    j = 0
    while j < len(cluster_pred):
        #print(str(j) + cluster_pred[j, 1])
        if kmer_similarity(top_kmer, cluster_pred[j, 1]) > 1: #can change similarity calc
            cluster.append(cluster_pred[j, 1])
            cluster_pred = np.delete(cluster_pred, j, 0)
            j = j-1
            #b = True
        j += 1
    if len(cluster) > 1:
        all_clusters2.append(cluster)


# In[123]:

all_clusters2


# In[22]:

# Prints clusters
for i in range(num_clusters):
    print('Cluster ' + str(i+1) + ':')
    print('\n'.join(all_clusters[i]))
    print('\n')


# ### Calculating the PWM

# In[21]:

# PARAMETERS

n = 250 # Number of patients (to train on)
m = 7 # Max number of kmers per patient
p = 2 # Number of positive kmers
k = 4 # Kmer size
num_test_cases = 200
num_top = 10 # Number of top test cases to select for seq logo clustering


# In[22]:

# Getting training data
xs, cs, ys, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size = k, num_pos_kmers=p)
print('Train data', xs.shape, cs.shape, ys.shape)

# Getting testing data based on positive kmer
XS, CS, YS, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size = k, num_pos_kmers=p,
                                           positive=ref)
print('Test data', XS.shape, CS.shape, YS.shape)

# Creating and training the model
model = MaxSnippetModel()
model.train(xs.reshape(n,m,-1), cs, ys, save_log=SL, recover=RE);


# In[25]:

# Generates set of random kmers and calculates the prediction score for each

def test_model(t):
## t = number of test cases
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

        #computing prediction value
        pred_value = sigmoid(np.average(model.predict(x.reshape(n,m,-1), CS)[0, :]));

        predictions.append((pred_value, s))
        
        
    # Adds positive kmers
    for i in range(len(ref)):
        poskmer = ref[i]
        # Encodes kmer string into binary input value for prediction
        poskmer_bin = kmer_to_binary(poskmer)
        poskmer_bin = np.expand_dims(np.expand_dims(poskmer_bin, axis=0), axis = 0)
        poskmer_bin = np.tile(poskmer_bin, (n, m, 1, 1))

        # Calculates prediction score
        pred_value = sigmoid(np.average(model.predict(XS.reshape(n,m,-1), CS)[0]))
        predictions.append((str(pred_value), poskmer))
        
        
    # Sorts kmers based on prediction score and selects top kmers
    predarray = np.asarray(predictions)
    ind = np.lexsort((predarray[:,1], predarray[:,0]))    
    a = predarray[ind]
    high_predictions = a[-num_top:, :] # selects topkmers for sequence logo

    high_predictions = high_predictions[::-1]
    high_predictions = a[-num_top:, :]
    high_predictions = high_predictions[::-1]
    return high_predictions


# In[ ]:

prediction = test_model(num_test_cases)


# In[ ]:



