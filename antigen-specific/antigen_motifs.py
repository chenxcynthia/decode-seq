
# coding: utf-8

# # Deep learning models for BCR repretoires in TCGA samples

# ## Load required packages and functions
# - Tensorflow package
# - Sklearn package
# - Functions in tcga_bcr.py
# 

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss

from tcga_bcr import *
from tcga_bcr_comp import plot_figures, plot_roc


# ## Load required data
# - Meta data for TCGA samples
# - Case name
# - Train and test datasets
# - Model and model parameters

# In[2]:

case = 'TCGA_Seq30Min300'
#para = (1, 10, 10, 3, 5)
para = (200, 1000, 500, 6, 10)

model_path = os.path.join('../work/', case+'-%s-%s-%s-%s-%s'%para)

num_motifs, batch_size, max_num_kmer, kmer_size, encode_size = para
meta = read_meta('../data/TCGA/tcga_clinical_CC.csv')
data_info, train_data, test_data = load_data(meta, case)
tumors = data_info.tumors.tolist()
data_info


# ## Base-model performance

# In[3]:

model_name = 'GeneSwitchModel'
resfile = os.path.join(model_path, model_name+'_result.csv')
res = pd.read_csv(resfile)
res.head()


# In[4]:

plot_figures(model_path=model_path, model_name=model_name, res=res, case=case)


# ## Build a transfer learning model
# 
# ### Train on a toy set

# In[5]:

target_gene = 'SCGB2A1'

new_name = 'TransferExpression_' + target_gene
labels = setup_expression(meta, tumors, read_expression([target_gene]))
features = format_trim_dims
encode = encode_aa_seq_index
count = encode_count_genes
model = DeepBCR(num_motifs=num_motifs, num_labels=1, encode_init=encode_size, save_path=model_path, model_name=new_name)

batch = next(get_next_batch(train_data, labels, size=100, kmer=kmer_size)) ## toy data
xs, cs, ys = format_trim_dims(batch, dim=max_num_kmer, encode=encode, count=count)
model.load_hidden_layer_parameters(model_path+'/GeneSwitchModel-par/model-3000', xs, cs, ys)
model.only_update_output_layer = True

train_iter = sample_data_iter(train_data, batch_size, kmer_size, max_num_kmer, labels, features, encode, count)
train = model.train_batch(train_iter, max_iterations=100, it_step_size=20)


# ### Predict on the test data

# In[6]:

model = DeepBCR(num_motifs=num_motifs, num_labels=1, encode_init=encode_size, model_name=new_name, save_path=model_path)
data = next(get_next_batch(test_data, labels, size=100, kmer=kmer_size, return_idx=True))
xs_data, cs_data, ys_real, ys_index = data

print(len(ys_real), ys_real[:2])
print(len(ys_index), ys_index[:2])


# ## Show the correlation for one gene

# In[7]:

new_name = 'TransferExpression_SCGB2A1'

train_real = np.loadtxt(os.path.join(model_path, new_name+'-res/train_real.txt'), dtype=str)
train_pred = np.loadtxt(os.path.join(model_path, new_name+'-res/train_3000.txt'))
ax = sns.jointplot(np.array(train_real[:,1], dtype='float'), train_pred)
ax.set_axis_labels("Gene expression", "Predicted")

test_real = np.loadtxt(os.path.join(model_path, new_name+'-res/test_real.txt'), dtype=str)
test_pred = np.loadtxt(os.path.join(model_path, new_name+'-res/test_3000.txt'))
ax = sns.jointplot(np.array(test_real[:,1], dtype='float'), test_pred)
ax.set_axis_labels("Gene expression", "Predicted")


# ## Compare to the result in the multiple linear regression

# In[8]:

new_name = 'TransferExpression_All'

gene_list = np.loadtxt(os.path.join(model_path, new_name+'-res/gene_list.txt'), dtype=str).tolist()
all_train_real = np.loadtxt(os.path.join(model_path, new_name+'-res/train_real.txt.gz'), dtype=str)
all_train_pred = np.loadtxt(os.path.join(model_path, new_name+'-res/train_3000.txt.gz'))

print(len(gene_list), all_train_real.shape, all_train_pred.shape)


# In[9]:

gene_idx = gene_list.index('SCGB2A1')

v1 = np.array(train_real[:,1], dtype='float')
v2 = np.array(all_train_real[:,(gene_idx+1)], dtype='float')

ax = sns.jointplot(v1, v2)
ax.set_axis_labels("Case 1", "Case 2")


# In[10]:

v1 = train_pred
v2 = all_train_pred[:,gene_idx]

ax = sns.jointplot(v1, v2)
ax.set_axis_labels("Case 1", "Case 2")


# In[11]:



