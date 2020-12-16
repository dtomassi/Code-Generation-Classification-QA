from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
import csv
from pathlib import Path

### Load in Model ###
p = str(Path.cwd())
glove_f_co = datapath(p + '/current-glove-vectors-mincount0.txt')
tmp_f_co = get_tmpfile(p + '/current-glove-vectors-mincount0-as-w2v.txt')

_ = glove2word2vec(glove_f_co, tmp_f_co)

co_model_g = KeyedVectors.load_word2vec_format(tmp_f_co)

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
