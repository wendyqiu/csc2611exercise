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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

NUM_TOP_WORDS = 5000

# # # # # # # # # # # # # # # # # # # #  # # # # # helper functions # # # # # # # # # # # # # # # # # # # # # # # # #


def ppmi(M1):
    col_totals = M1.sum(axis=0)
    total = col_totals.sum()
    row_totals = M1.sum(axis=1)
    denominator = np.outer(row_totals, col_totals) / total
    with np.errstate(divide='ignore', invalid='ignore'):
        M1 = M1 / denominator
        M1 = np.log(M1)
    M1[np.isinf(M1)] = 0.0  # set log(0) = 0
    M1[M1 < 0] = 0.0  # set all negatives to 0
    return M1

# # # # # # # # # # # # # # # # # # # # #  # # # # # helpers end # # # # # # # # # # # # # # # # # # # # # # # # # #


word_pair_score = [{'pair': ["cord", "smile"], 'score': 0.02},
                   {'pair': ["hill", "woodland"], 'score': 1.48},
                   {'pair': ["rooster", "voyage"], 'score': 0.04},
                   {'pair': ["car", "journey"], 'score': 1.55},
                   {'pair': ["noon", "string"], 'score': 0.04},
                   {'pair': ["cemetery", "mound"], 'score': 1.69},
                   {'pair': ["fruit", "furnace"], 'score': 0.05},
                   {'pair': ["glass", "jewel"], 'score': 1.78},
                   {'pair': ["autograph", "shore"], 'score': 0.06},
                   {'pair': ["magician", "oracle"], 'score': 1.82},
                   {'pair': ["automobile", "wizard"], 'score': 0.11},
                   {'pair': ["crane", "implement"], 'score': 2.37},
                   {'pair': ["mound", "stove"], 'score': 0.14},
                   {'pair': ["brother", "lad"], 'score': 2.41},
                   {'pair': ["grin", "implement"], 'score': 0.18},
                   {'pair': ["sage", "wizard"], 'score': 2.46},
                   {'pair': ["asylum", "fruit"], 'score': 0.19},
                   {'pair': ["oracle", "sage"], 'score': 2.61},
                   {'pair': ["asylum", "monk"], 'score': 0.39},
                   {'pair': ["bird", "crane"], 'score': 2.63},
                   {'pair': ["graveyard", "madhouse"], 'score': 0.42},
                   {'pair': ["bird", "cock"], 'score': 2.63},
                   {'pair': ["glass", "magician"], 'score': 0.44},
                   {'pair': ["food", "fruit"], 'score': 2.69},
                   {'pair': ["boy", "rooster"], 'score': 0.44},
                   {'pair': ["brother", "monk"], 'score': 2.74},
                   {'pair': ["cushion", "jewel"], 'score': 0.45},
                   {'pair': ["asylum", "madhouse"], 'score':  3.04},
                   {'pair': ["monk", "slave"], 'score': 0.57},
                   {'pair': ["furnace", "stove"], 'score': 3.11},
                   {'pair': ["asylum", "cemetery"], 'score': 0.79},
                   {'pair': ["magician", "wizard"], 'score': 3.21},
                   {'pair': ["coast", "forest"], 'score': 0.85},
                   {'pair': ["hill", "mound"], 'score': 3.29},
                   {'pair': ["grin", "lad"], 'score': 0.88},
                   {'pair': ["cord", "string"], 'score': 3.41},
                   {'pair': ["shore", "woodland"], 'score': 0.90},
                   {'pair': ["glass", "tumbler"], 'score': 3.45},
                   {'pair': ["monk", "oracle"], 'score': 0.91},
                   {'pair': ["grin", "smile"], 'score': 3.46},
                   {'pair': ["boy", "sage"], 'score': 0.96},
                   {'pair': ["serf", "slave"], 'score': 3.46},
                   {'pair': ["automobile", "cushion"], 'score': 0.97},
                   {'pair': ["journey", "voyage"], 'score': 3.58},
                   {'pair': ["mound", "shore"], 'score': 0.97},
                   {'pair': ["autograph", "signature"], 'score': 3.59},
                   {'pair': ["lad", "wizard"], 'score': 0.99},
                   {'pair': ["coast", "shore"], 'score': 3.6},
                   {'pair': ["forest", "graveyard"], 'score': 1.0},
                   {'pair': ["forest", "woodland"], 'score': 3.65},
                   {'pair': ["food", "rooster"], 'score': 1.09},
                   {'pair': ["implement", "tool"], 'score': 3.66},
                   {'pair': ["cemetery", "woodland"], 'score': 1.18},
                   {'pair': ["cock", "rooster"], 'score': 3.68},
                   {'pair': ["shore", "voyage"], 'score': 1.22},
                   {'pair': ["boy", "lad"], 'score': 3.82},
                   {'pair': ["bird", "woodland"], 'score': 1.24},
                   {'pair': ["cushion", "pillow"], 'score': 3.84},
                   {'pair': ["coast", "hill"], 'score':1.26},
                   {'pair': ["cemetery", "graveyard"], 'score': 3.88},
                   {'pair': ["furnace", "implement"], 'score': 1.37},
                   {'pair': ["automobile", "car"], 'score': 3.92},
                   {'pair': ["crane", "rooster"], 'score': 1.41},
                   {'pair': ["midday", "moon"], 'score': 3.94},
                   {'pair': ["gem", "jewel"], 'score': 3.94}]

P_word_pairs = [each_pair['pair'] for each_pair in word_pair_score]
print("len of P_word_pairs: " + str(len(P_word_pairs)))

table_word_set = set([word for word_list in P_word_pairs for word in word_list])

punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
               '@', '[', '\\', ']',
               '^', '_', '`', '{', '|', '}', '~', '``', "''", '--']

# step 2: Extract the 5000 most common English words
all_texts = brown.words()
punctuation_stopwords = punctuation + stopwords.words('english')
plain_words = [word for word in all_texts if word not in punctuation_stopwords]  # filter out stop words and punctuation
filter_words = list(filter(lambda x: x.isalpha() and len(x) > 1, plain_words))  # remove numbers and single letter words
fdist = nltk.FreqDist(w.lower() for w in filter_words)

brown_common_tuple = fdist.most_common(NUM_TOP_WORDS)  # extract the top 5000 frequently used words
brown_common_words = set([word_tuple[0] for word_tuple in brown_common_tuple])

W_combined_word_set = brown_common_words | table_word_set
W_len = len(W_combined_word_set)
print("step 2 - total number of words in the combined word set: {}".format(W_len))

# step 3: Construct a word-context vector model using bigram counts, denoted by M1
n = 2
bigram_full = ngrams(brown.words(), n)
bigram_counts = Counter(bigram_full)

M1_dense = np.zeros(shape=(W_len, W_len))
combined_word_list = list(W_combined_word_set)
for word_row in combined_word_list:
    row_idx = combined_word_list.index(word_row)
    for word_col in combined_word_list:
        col_idx = combined_word_list.index(word_col)
        curr_freq = bigram_counts[(word_row, word_col)]
        if curr_freq != 0:  # here we use co-occurrence with order
            M1_dense[row_idx][col_idx] = curr_freq

# Step 4: Compute positive pointwise mutual information on M1. Denote this model as M1+

M1_plus_dense = ppmi(M1_dense)

# Step 5: Construct a latent semantic model, denoted by M2
M1_plus_dense[np.isnan(M1_plus_dense)] = 0
M1_plus = csr_matrix(M1_plus_dense)

u, s, vh = np.linalg.svd(M1_plus_dense, full_matrices=True)
print("shape of u, s, v: {}, {}, {}".format(u.shape, s.shape, vh.shape))
smat = np.diag(s)
M2_full = np.dot(u, np.dot(smat, vh))
print("shape of M2_full: {}".format(M2_full.shape))
M2_10 = M2_full[:,:10]
M2_100 = M2_full[:,:100]
M2_300 = M2_full[:,:300]
print("after shape of M2_10, M2_100, M2_300: {}, {}, {}".format(M2_10.shape, M2_100.shape, M2_300.shape))

# Step 6: Find all pairs of words in Table 1 of RG65 that are also available in W. Denote these pairs as P.
# Record the human-judged similarities of these word pairs from the table and denote similarity values as S.
S_word_pair_score = [each_dict['score'] for each_dict in word_pair_score]

# Step 7: Calculate cosine similarity between each pair of words in P: S_M1, S_M2_10 , S_M2_100 , S_M2_300
similarity_scores = []
for each_pair in P_word_pairs:
    a_idx = combined_word_list.index(each_pair[0])
    b_idx = combined_word_list.index(each_pair[1])

    S_M1 = cosine_similarity(np.array(M1_dense[a_idx]).reshape(1, -1), np.array(M1_dense[b_idx]).reshape(1, -1))[0][0]
    S_M1_plus = cosine_similarity(np.array(M1_plus_dense[a_idx]).reshape(1, -1),
                                  np.array(M1_plus_dense[b_idx]).reshape(1, -1))[0][0]
    S_M2_10 = cosine_similarity(np.array(M2_10[a_idx]).reshape(1, -1), np.array(M2_10[b_idx]).reshape(1, -1))[0][0]
    S_M2_100 = cosine_similarity(np.array(M2_100[a_idx]).reshape(1, -1), np.array(M2_100[b_idx]).reshape(1, -1))[0][0]
    S_M2_300 = cosine_similarity(np.array(M2_300[a_idx]).reshape(1, -1), np.array(M2_300[b_idx]).reshape(1, -1))[0][0]
    curr_similarity = {'pair': each_pair,
                       'S_M1': S_M1, 'S_M1_plus': S_M1_plus, 'S_M2_10': S_M2_10, 'S_M2_100': S_M2_100, 'S_M2_300': S_M2_300}
    similarity_scores.append(curr_similarity)

# Step 8: Report Pearson correlation between S and each of the model-predicted similarities
S_M1_full = []
S_M2_10_full = []
S_M2_100_full = []
S_M2_300_full = []
S_M1_plus_full = []
for each_sim in similarity_scores:
    S_M1_full.append(each_sim['S_M1'])
    S_M1_plus_full.append(each_sim['S_M1_plus'])
    S_M2_10_full.append(each_sim['S_M2_10'])
    S_M2_100_full.append(each_sim['S_M2_100'])
    S_M2_300_full.append(each_sim['S_M2_300'])

corr_M1, _ = pearsonr(np.array(S_M1_full), np.array(S_word_pair_score))
print('Pearsons correlation between S and S_M1: {0}'.format(corr_M1))
corr_M1_plus, _ = pearsonr(S_M1_plus_full, S_word_pair_score)
print('Pearsons correlation between S and S_M1+: {0}'.format(corr_M1_plus))
corr_M2_10, _ = pearsonr(S_M2_10_full, S_word_pair_score)
print('Pearsons correlation between S and S_M2_10: {0}'.format(corr_M2_10))
corr_M2_100, _ = pearsonr(S_M2_100_full, S_word_pair_score)
print('Pearsons correlation between S and S_M2_100: {0}'.format(corr_M2_100))
corr_M2_300, _ = pearsonr(S_M2_300_full, S_word_pair_score)
print('Pearsons correlation between S and S_M2_300: {0}'.format(corr_M2_300))
