
# coding: utf-8

# # Determining binding site of CDR3

# ### Load required packages and functions

# In[7]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss

from tcga_bcr import *
from deep_bcr import *
from seq_logo_functions import *
from tcga_bcr_comp import plot_figures, plot_roc


# ### Load required data

# In[5]:

case = 'Tophat_Seq30Min300'
para = (200, 1000, 500, 6, 20)

num_motifs, batch_size, max_num_kmer, kmer_size, encode_size = para


# In[6]:

meta = read_meta('../data/TCGA/tcga_clinical_CC.csv')
data_info, train_data, test_data = load_data(meta, case)
tumors = data_info.tumors.tolist()
data_info


# ### Initialize model

# In[8]:

labels = setup_tumor_types(meta, tumors)
features = format_balance_labels
encode = encode_aa_seq_binary_ext_dim
count = encode_count_genes

model_path = os.path.join('../work/', case+'-%s-%s-%s-%s-%s'%para)
model = GeneSwitchModelFast(num_motifs=num_motifs, num_labels=len(tumors), encode_init=encode_size, 
                                   save_path=model_path)


# ### Determine most likely binding region in CDR3
# 
# - Find all relevant CDR3 kmers (same length, complete)
# - Using sliding window method, find prediction scores of window kmers
# - Generate comprehensive sequence logo 

# In[9]:

# Reads cdr3 gene sequence file
file = open("/Users/cynthiachen/Documents/Research/Internship-2018/code/DeepBCR/data/TCGA/complete_cdr3s.txt")
complete_cdr3_list = file.read().splitlines() 


# In[10]:

# Creates list of CDR3s that are a certain length
cdr3 = []
cdr3_length = 16
for i in complete_cdr3_list:
    if (len(i) == cdr3_length):
        cdr3.append(i)


# In[11]:

def kmer_to_binary(kmer, aa_list='ACDEFGHIKLMNPQRSTVWY'):
    binary = []
    for i in range(len(kmer)):
        char = kmer[i]
        index = aa_list.find(char)
        array = np.zeros(len(aa_list))
        array[index] = 1
        binary.append(array)
    return binary


# In[12]:

kmer_length = 6
num_labels = len(tumors)
all_pred = []

r = 3 #len(cdr3)
for i in range(r):
    sequence = cdr3[i]
    x = []
    windows = cdr3_length - kmer_length + 1
    sequence_pred = []
    for j in range(windows):
        kmer = sequence[j:(j+kmer_length)]
        x.append(kmer_to_binary(kmer))
    x = np.expand_dims(np.asarray(x), axis = 1)
    CS = np.ones((windows, 1, 5))
    ys = np.ones((windows, num_labels))
    if i == 0:
        model.load(x, CS, ys)
    pred = model.predict(x, CS)#[0, :] # prediction array
    max_pred = []
    for row in pred:
        max_pred.append(sigmoid(np.amax(row)))
    all_pred.append(max_pred)


# In[14]:

# Simple visualization of prediction score for each of the "windows"

from matplotlib.pyplot import figure


x = range(windows)
for i in range(r):
    plt.subplot(1, r, i+1)
    y = all_pred[i]
    plt.ylim([min(y)-0.03, max(y)+0.005])
    plt.bar(x, y, width = 0.6)
    plt.title('CDR3_' + str(i+1), fontsize = 14)
plt.subplots_adjust(wspace = 0.9, top=0.8)
plt.suptitle('Prediction scores for each window', fontsize = 18)
plt.show()


# In[15]:

# fig, ax = plt.subplots(1, 3, sharey = True, figsize = (30, 6))
# ax[0].ylim([min(y)-0.03, max(y)])
# ax[0].bar(x, y, width = 0.5)


# Calculate binding scores for each of the positions in a certain CDR3

# In[16]:

binding_sum = np.zeros((r, cdr3_length))
binding_count = np.zeros((r, cdr3_length))
binding = np.zeros((r, cdr3_length))
for i in range(r):
    for j in range(cdr3_length):
        for k in range(windows):
            if (j >= k and j <= k+5):
                binding_sum[i][j] += all_pred[i][k]
                binding_count[i][j] += 1

for i in range(r):
    for j in range(cdr3_length):
        binding[i][j] = binding_sum[i][j] / binding_count[i][j]


# In[17]:

x = range(cdr3_length)
for i in range(r):
    plt.subplot(1, r, i+1)
    y = binding[i]
    plt.ylim([min(y)-0.03, max(y)+0.005])
    plt.bar(x, y, width = 0.5)
    plt.title('CDR3_' + str(i+1), fontsize = 14)
plt.subplots_adjust(wspace = 0.9, top=0.8)
plt.suptitle('Binding affinity for each position', fontsize = 18, y =1)
plt.show()


# ### Calculate PWM

# In[18]:

aa_list='ACDEFGHIKLMNPQRSTVWY'
pwm = np.zeros((len(aa_list), cdr3_length))
        
for i in range(cdr3_length):
    for j in range(len(aa_list)):
        aa = aa_list[j]
        top = 0
        total = 0
        for k in range(r):
            #print(cdr3[k])
            if(cdr3[k][i] == aa):
                top += binding[k][i]
            total += binding[k][i]
        pwm[j][i] = round(top / total, 3) 


# In[19]:

# Create protein format pwm text file for motifStack usage
f = open('pwm.txt', 'w')
f.write('\t')
for i in range(20):
    f.write(aa_list[i] + ' ')
f.write('\n')
for i in range(cdr3_length):
    f.write(str(i+1) + ' \t')
    for j in range(len(aa_list)):
        f.write(str(pwm[j][i]) + ' ')
    f.write('\n')
f.close()


# ### Dictionary of unique kmers + prediction scores for determining CDR3 binding site 

# In[20]:

# Randomly shuffle cdr3 to get a better sampling
from random import shuffle
shuffle(cdr3)


# In[21]:

x_dict = []
r_new=10000
for i in range(r_new):
    sequence = cdr3[i]
    x = []
    windows = cdr3_length - kmer_length + 1
    sequence_pred = []
    for j in range(windows):
        kmer = sequence[j:(j+kmer_length)]
        x_dict.append(kmer)
        
x_dict = list(set(x_dict))
x_dict_bin = []
for kmer in x_dict:
    x_dict_bin.append(kmer_to_binary(kmer))


# In[22]:

n_unique = len(x_dict_bin)
x = np.expand_dims(np.asarray(x_dict_bin), axis = 1)
CS = np.ones((n_unique, 1, 5))
ys = np.ones((n_unique, num_labels))
model.load(x, CS, ys)
pred = model.predict(x, CS)


# In[23]:

# Dictionary (kmer string -> prediction array)
d = dict(zip(x_dict, pred)) 


# In[24]:

len(d)


# In[25]:

# Calculate binding scores
binding_sum = np.zeros((num_labels, r_new, cdr3_length))
binding_count = np.zeros((num_labels, r_new, cdr3_length))
for c in range(num_labels):
    for i in range(r_new):
        for j in range(cdr3_length):
            for k in range(windows):
                if (j >= k and j <= k+5):
                    binding_sum[c][i][j] += sigmoid(d[cdr3[i][k:(k+6)]][c])
                    binding_count[c][i][j] += 1
                    
binding = np.divide(binding_sum, binding_count) # element-wise division


# In[26]:

# For each CDR3 position, taking the average binding score among all CDR3s

binding_avg = np.average(binding, axis = 1)

x = range(cdr3_length)
fig = plt.figure(figsize = (20, 50))
for i in range(num_labels):
    plt.subplot(num_labels, 5, i+1)
    y = binding_avg[i]
    plt.ylim([min(y)-0.005, max(y)+0.005])
    barlist = plt.bar(x, y, width = 0.4)
#     for j in range(len(barlist)):
#         barlist[j].set_color(binding_avg[i][j])
    plt.title('Cancer type: ' + tumors[i], fontsize = 14)
plt.subplots_adjust(hspace = 0.5, wspace = 0.5, top=0.8)
#plt.suptitle('Binding affinity for each position', fontsize = 18, y =1)
plt.show()


# In[27]:

binding_arr = np.zeros((num_labels, r_new))
top_ind = []
for i in range(num_labels):
    for j in range(r_new):
        binding_arr[i][j] = np.amax(binding[i][j])
    predarray = binding_arr[i]
    ind = np.argsort((predarray))
    top_ind.append(ind)


# In[28]:

np.asarray(top_ind).shape


# In[29]:

# top = 100
# pwm_dict = np.zeros((num_labels, len(aa_list), cdr3_length))

# binding = np.asarray(binding)
# for i in range(num_labels):
#     top_cdr3 = []
#     for j in range(top):
#         index = top_ind[i][j]
#         top_cdr3.append(binding[i][index][:])


# In[30]:

aa_list='ACDEFGHIKLMNPQRSTVWY'
pwm_top = np.zeros((num_labels, len(aa_list), cdr3_length))

top = 20
for c in range(num_labels):
    for i in range(cdr3_length):
        for j in range(len(aa_list)):
            aa = aa_list[j]
            numerator = 0
            total = 0
            for k in range(top):
                index = top_ind[c][k]
                if(cdr3[k][i] == aa):
                    numerator += binding[c][index][i]
                total += binding[c][index][i]
            pwm_top[c][j][i] = round(numerator / total, 3) 


# In[31]:

# # Create PWM file for all cancer types combined
# aa_list='ACDEFGHIKLMNPQRSTVWY'
# pwm_dict = np.zeros((num_labels, len(aa_list), cdr3_length))
        
# for c in range(num_labels):
#     for i in range(cdr3_length):
#         for j in range(len(aa_list)):
#             aa = aa_list[j]
#             top = 0
#             total = 0
#             for k in range(r_new):
#                 if(cdr3[k][i] == aa):
#                     top += binding[c][k][i]
#                 total += binding[c][k][i]
#             pwm_dict[c][j][i] = round(top / total, 3) 


# In[32]:

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


# ### Combining kmer window scores

# In[55]:

x = np.expand_dims(np.asarray(windows), axis = 0)
CS = np.ones((1, num_windows, 5))
ys = np.ones((1, num_labels))
model.load(x, CS, ys)
pred = model.predict(x, CS)


# In[130]:

all_windows = []
r = 10000
for i in range(r):
    sample_cdr3 = cdr3[i]
    windows = []
    for j in range(num_windows):
        windows.append(kmer_to_binary(sample_cdr3[j:j+kmer_length]))
    all_windows.append(windows)


# In[131]:

num_windows = cdr3_length - kmer_length + 1
CS = np.ones((r, num_windows, 5))
ys = np.ones((r, num_labels))
model.load(np.zeros((1, num_windows, kmer_length, len(aa_list))), CS, ys)

all_pred = model.predict(np.asarray(all_windows), CS)


# In[133]:

high_predictions_all = []
num_top = 20
for i in range(num_labels):
    predarray = np.transpose(all_pred)[i]
    ind = np.argsort(predarray)
    #sorted_predarray = np.sort(predarray)
    #high_predictions = sorted_predarray[-num_top:, :] # selects topkmers for sequence logo
    high_pred = np.asarray(cdr3)[ind[0:num_top]]
    high_predictions_all.append(high_pred)


# In[134]:

# Prints motifs
for i in range(num_labels):
    print(tumors[i] + ':')
    print('\n'.join(high_predictions_all[i]))
    print('\n')


# In[140]:

plt.plot(all_pred[0])


# In[151]:

average_scores = []
for i in range(num_labels):
    average_scores.append(np.average(np.transpose(all_pred)[i]))


# In[154]:

plt.hist(np.transpose(all_pred)[0])


# In[155]:

plt.hist(np.transpose(all_pred)[4])


# In[152]:

plt.plot(average_scores)


# In[120]:

sample.append(high_pred.tolist())


# In[113]:

np.asarray(cdr3)[a[0:num_top]]


# In[101]:

a = np.argsort(predarray)


# In[112]:

a[0:num_top]


# In[86]:

all_pred[0]


# In[ ]:




# In[72]:

all_pred = []
for i in range(num_labels):
    for j in range(r):
        

