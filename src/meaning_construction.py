"""
CSC2611 Exercise
Construct various language models and investigate word pair similarity
Wendy Qiu 2021.02.01
"""

import nltk
from nltk.corpus import brown, stopwords
from nltk.util import ngrams
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix


# # # # # # # # # # # # # # # # # # # #  # # # # # helper functions # # # # # # # # # # # # # # # # # # # # # # # # #
def pmi(M1, positive=True):
    col_totals = M1.sum(axis=0)
    total = col_totals.sum()
    row_totals = M1.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    M1 = M1 / expected
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        M1 = np.log(M1)
    M1[np.isinf(M1)] = 0.0  # log(0) = 0
    if positive:
        M1[M1 < 0] = 0.0
    return M1
# # # # # # # # # # # # # # # # # # # # #  # # # # # helpers end # # # # # # # # # # # # # # # # # # # # # # # # # #

word_pairs = [["cord", "smile"], ["hill", "woodland"],
              ["rooster", "voyage"], ["car", "journey"],
              ["noon", "string"], ["cemetery", "mound"],
              ["fruit", "furnace"], ["glass", "jewel"],
              ["autograph", "shore"], ["magician", "oracle"],
              ["automobile", "wizard"], ["crane", "implement"],
              ["mound", "stove"], ["brother", "lad"],
              ["grin", "implement"], ["sage", "wizard"],
              ["asylum", "fruit"], ["oracle", "sage"],
              ["asylum", "monk"], ["bird", "crane"],
              ["graveyard", "madhouse"], ["bird", "cock"],
              ["glass", "magician"], ["food", "fruit"],
              ["boy", "rooster"], ["brother", "monk"],
              ["cushion", "jewel"], ["asylum", "madhouse"],
              ["monk", "slave"], ["furnace", "stove"],
              ["asylum", "cemetery"], ["magician", "wizard"],
              ["coast", "forest"], ["hill", "mound"],
              ["grin", "lad"], ["cord", "string"],
              ["shore", "woodland"], ["glass", "tumbler"],
              ["monk", "oracle"], ["grin", "smile"],
              ["boy", "sage"], ["serf", "slave"],
              ["automobile", "cushion"], ["journey", "voyage"],
              ["mound", "shore"], ["autograph", "signature"],
              ["lad", "wizard"], ["forest", "woodland"],
              ["forest", "graveyard"], ["implement", "tool"],
              ["food", "rooster"], ["cock", "rooster"],
              ["cemetery", "woodland"], ["boy", "lad"],
              ["shore", "voyage"], ["cushion", "pillow"],
              ["bird", "woodland"], ["cemetery", "graveyard"],
              ["coast", "hill"], ["automobile", "car"],
              ["furnace", "implement"], ["midday", "moon"],
              ["crane", "rooster"],  ["coast", "shore"], ["gem", "jewel"]]

table_word_set = set([word for word_list in word_pairs for word in word_list])

punctuation = ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']',
               '^','_','`','{','|','}','~','``',"''",'--']

# step 2: Extract the 5000 most common English words
all_texts = brown.words()
punctuation_stopwords = punctuation + stopwords.words('english')
plain_words = [word for word in all_texts if word not in punctuation_stopwords] # filter out stop words and punctuation
filter_words = list(filter(lambda x: x.isalpha() and len(x) > 1, plain_words))  # remove numbers and single letter words
fdist = nltk.FreqDist(w.lower() for w in filter_words)

brown_common_tuple = fdist.most_common(5000)    # extract the top 5000 frequently used words
brown_common_words = set([word_tuple[0] for word_tuple in brown_common_tuple])

combined_word_set = brown_common_words | table_word_set
W_len = len(combined_word_set)
print("step 2 - total number of words in the combined word set: {}".format(W_len))

# step 3: Construct a word-context vector model using bigram counts, denoted by M1
n = 2
bigram_full = ngrams(brown.words(), n)
bigram_freq = Counter(bigram_full)

M1_dense = np.empty(shape=(W_len, W_len))
combined_word_list = list(combined_word_set)
for word_in_row in combined_word_list:
    col_idx = combined_word_list.index(word_in_row)
    for word_in_col in combined_word_list:
        row_idx = combined_word_list.index(word_in_col)
        curr_freq = bigram_freq[(word_in_row, word_in_col)]
        if curr_freq != 0:
            M1_dense[row_idx][col_idx] = curr_freq

M1_sparse = csr_matrix(M1_dense)
print(M1_sparse)

# Step 4: Compute positive pointwise mutual information on M1. Denote this model as M1+
M_plus = pmi(M1_sparse, positive=True)

# Step 5: Construct a latent semantic model, denoted by M2

