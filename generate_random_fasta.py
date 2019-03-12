
# coding: utf-8

# In[20]:

from random import *
peptide_char = ['A', 'R', 'N', 'D', 'C', 'H', 'I', 'Q', 'E', 'G', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                'W', 'Y', 'V'];
def generate_cdr3(length):
    peptide = ''
    for i in range(length):
        peptide += peptide_char[int(random()*20)]
    return peptide


# In[21]:

generate_cdr3(5)


# In[24]:

ofile = open("small_fasta.txt", "w")

num_seq = 300000
seq_length = 16
for i in range(num_seq):
    ofile.write(">" + str(i) + "\n" + generate_cdr3(seq_length) + "\n")
#do not forget to close it
ofile.close()


# In[ ]:



