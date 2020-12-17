from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot
import re

#### Open File ###
tfile = open('fresh-combined-datasetVocab.txt','r')

token_words = []

for l in tfile:
    bracket_re = re.compile(r'\[(.*)\]')
    tl = bracket_re.search(l)

    q_re = re.compile(r'\'(.*?)\'')
    toli = q_re.findall(tl.group(1))
    token_words.append(toli)

### Word2Vec ###
sentences = token_words

model = Word2Vec(sentences, window=15, min_count=0, sg=1, iter=10)
print('\n\nModel Summary (sg + n):')
print(model)
model.wv.save('current-w2v-model-mincount0.bin')
