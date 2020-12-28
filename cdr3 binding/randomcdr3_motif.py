
# coding: utf-8

# Data import - use after scores for each CDR3 have been calculated

# In[21]:

import pandas as pd
import numpy as np

scores = pd.DataFrame()
curr = pd.read_csv('/Users/cynthiachen/Documents/Research/Internship-2018/code/DeepBCR/transfer_learning/work/testfasta.csv', sep=',')
scores = scores.append(curr)

from Bio import SeqIO
cdr3s = []
fasta_sequences = SeqIO.parse(open('/Users/cynthiachen/Documents/Research/Internship-2018/code/DeepBCR/transfer_learning/src/small_fasta.txt'),'fasta')

for fasta in fasta_sequences:
    name, sequence = fasta.id, fasta.seq.tostring()
    cdr3s.append(sequence)

combined = [list(a) for a in zip(cdr3s, scores['WT1'])]


# In[59]:

both = pd.DataFrame()
curr = pd.read_csv('/Users/cynthiachen/Documents/Research/Internship-2018/code/DeepBCR/transfer_learning/work/testfasta.csv', sep=',')
both = both.append(curr)
combined = [list(a) for a in zip(both['Unnamed: 0'], both['WT1'])]


# Sorting

# In[60]:

num_top = int(len(cdr3s) * 0.0001)

ind = np.argsort(np.array(combined)[:, 1])
#sorted_predarray = np.sort(predarray)
#high_predictions = sorted_predarray[-num_top:, :] # selects topkmers for sequence logo
high_pred = np.asarray(combined)[ind[-num_top:]]


# In[61]:

# Prints motifs
print('\n'.join(np.ndarray.tolist(high_pred[:, 0])))


# In[ ]:

binding_arr = np.zeros((num_labels, r_new))
top_ind = []
for i in range(num_labels):
    for j in range(r_new):
        binding_arr[i][j] = np.amax(binding[i][j]
    predarray = binding_arr[i]
    ind = np.argsort((predarray))
    top_ind.append(ind)

