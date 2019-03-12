
# coding: utf-8

# # Rank CDR3s According to Predictions
# 
# ## Library and helper functions

# In[1]:

get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from sklearn.metrics import roc_curve, auc

def init_plotting(w,h):
    sns.set(style="ticks")
    plt.rcParams['figure.figsize'] = (w,h)
    plt.rcParams['font.size'] = 14
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

def append_cdr3(file, current=None):
    df = pd.read_csv(file, index_col=0)
    if current is None:
        return df
    assert df.index.equals(current.index)
    name = list(df)[0]
    current[name] = df[name]
    return current

def save_target(tcga, data, name, top=50):
    top_cdr3 = data.sort_values([name], ascending=False).index[:top]
    sel_cdr3 = tcga_bcr.loc[tcga_bcr['CDR3_aa'].isin(top_cdr3),]
    sel_cdr3.to_csv('../work/target_'+name+'_BCR.csv')
    return sel_cdr3

def add_pvalue(X, Y, data, ax=None, x1=0, x2=1):
    if ax is None:
        fig, ax = plt.figure()
    case1, case2 = data[X].unique().tolist()
    c1 = data.loc[data[X] == case1, Y].values
    c2 = data.loc[data[X] == case2, Y].values
    statistic, pvalue = ttest_ind(c1, c2, equal_var=False)
    if pvalue < 0.001:
        pv = 'P = %.1E'%pvalue
    else:
        pv = f'P = {pvalue:.1}'
    rg = data[Y].max() - data[Y].min()
    y, h, col = data[Y].max() + 0.03*rg, 0.03*rg, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, pv, ha='center', va='bottom', color=col)


# ## Prediction on know tumor antigens

# In[2]:

data1 = append_cdr3('../work/pred.AllCDR3_SCGB2A1_UCEC.csv.gz')
data1 = append_cdr3('../work/pred.AllCDR3_WT1_BRCA-LumA.csv.gz', data1)
#data1 = append_cdr3('../work/pred.Paired_Selected.csv.gz')

data1.head()


# In[3]:

init_plotting(4,4)
g = sns.jointplot('SCGB2A1', 'WT1', data=data1, kind="hex")
g.annotate(pearsonr)
plt.show()


# In[4]:

data1.sort_values(['SCGB2A1'], ascending=False).head()


# In[5]:

data1.sort_values(['WT1'], ascending=False).head()


# ## Search the top CDR3 in all TCGA data

# In[6]:

tcga_bcr = pd.read_csv('../data/TCGA/tcga_bcrh_v20180405.txt.gz', sep='\t')
tcga_bcr.head()


# In[7]:

sel_columns = ['TCGA_id', 'Disease', 'sample_type', 'CDR3_aa']
save_target(tcga_bcr, data1, 'SCGB2A1')[sel_columns].head()


# In[8]:

save_target(tcga_bcr, data1, 'WT1')[sel_columns].head()


# ## Prediction on CDR3s of known targets

# In[9]:

AFP = append_cdr3('../work/pred.AFP_AllExp.csv')
AFP = AFP[AFP.index.str.contains('IGH')]
AFP['Target'] = 'AFP'
WT1 = append_cdr3('../work/pred.WT1_AllExp.csv')
WT1['Target'] = 'WT1'
#WT1 = WT1[~WT1.index.str.contains('part')]

data2 = pd.concat([AFP, WT1])
data2 = data2[['Target','AFP','WT1','SCGB2A1']]
data2.head()


# In[10]:

init_plotting(8, 4)
fig, axs = plt.subplots(ncols=3)

sns.boxplot('Target', 'AFP', data=data2, ax=axs[0])
add_pvalue('Target', 'AFP', data=data2, ax=axs[0])
sns.boxplot('Target', 'WT1', data=data2, ax=axs[1])
add_pvalue('Target', 'WT1', data=data2, ax=axs[1])
sns.boxplot('Target', 'SCGB2A1', data=data2, ax=axs[2])
add_pvalue('Target', 'SCGB2A1', data=data2, ax=axs[2])

for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('')

plt.tight_layout()
plt.show()


# In[11]:

# Compute ROC curve
fpr, tpr, _ = roc_curve(data2['Target'] == 'WT1', data2['WT1'])
roc_auc = auc(fpr, tpr)


# In[12]:

init_plotting(5, 4)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='WT1 (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


# ## Prediction on CDR3s with paired heavy and light chains

# In[13]:

data3 = pd.read_csv('../work/pred.Paired_Selected.csv.gz', index_col=0)
data3['Target'] = 'Paired'
data3 = data3[list(data2)]
data3.head()


# In[14]:

init_plotting(4, 4)
sns.stripplot('Target', 'WT1', data=pd.concat([data2, data3]))
plt.show()


# In[15]:

data4 = pd.read_csv('../../PairedBCR/Paired_BCR_Chains.v20181120.csv.gz')
data4.head()


# In[17]:

top_WT1 = data3.sort_values('WT1', ascending=False).head()
print(top_WT1)
data4[data4['H_cdr3'].isin(top_WT1.index[:2])]


# In[18]:

top_SCG = data3.sort_values('SCGB2A1', ascending=False).head()
print(top_SCG)
data4[data4['H_cdr3'].isin(top_SCG.index[:2])]


# In[ ]:



