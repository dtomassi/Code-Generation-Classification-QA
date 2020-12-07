from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from pathlib import Path

### Load in Model ###
glove_f_csn = datapath(p + '/vectors-csn.txt')
tmp_f_csn = get_tmpfile(p + '/glove_w2v_csn.txt')

_ = glove2word2vec(glove_f_csn, tmp_f_csn)

csn_model_g = KeyedVectors.load_word2vec_format(tmp_f_csn)

### Similarity Tests ###
# Check how similar two input tokens are to each other
print('\nSimilarity Scores Between Words:')

w1 = 'print'
w2 = 'in'
print('%r\t%r\t%.2f' % (w1, w2, model.similarity(w1, w2)))

w1 = 'for'
w2 = 'in'
print('%r\t%r\t%.2f' % (w1, w2, model.similarity(w1, w2)))

w1 = 'if'
w2 = 'else'
print('%r\t%r\t%.2f' % (w1, w2, model.similarity(w1, w2)))

# Check most similar tokens to input
print('\nTokens Most Similar to For')
print(model.most_similar('for'))

print('\nTokens Most Similar to Import')
print(model.most_similar('import'))
