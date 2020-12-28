import numpy as np

def binary_to_kmer(binary, aa_list='ACDEFGHIKLMNPQRSTVWY'):
    kmer = []
    for i in range(len(binary)):
        for j in range(len(aa_list)):
            if(binary[i][j] == 1):
                kmer.append(aa_list[j])
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


import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def train_model():
    # Getting training data
    xs, cs, ys, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size = k, 
                                   num_pos_kmers=p)
    print('Train data', xs.shape, cs.shape, ys.shape)

    # Creating and training the model
    model = MaxSnippetModel()
    model.train(xs.reshape(n,m,-1), cs, ys, save_log=SL, recover=RE); # get weights??
    return model, ref


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


def kmer_similarity(kmer1, kmer2):
    score = 0
    for i in range(len(kmer1)):
        if kmer1[i] == kmer2[i]:
            score += 1
    return score

