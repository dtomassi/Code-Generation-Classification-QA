from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot
import re

#### Open File ###
tfile = open('combined-datasetVocab.txt','r')

token_words = []

for l in tfile:
    bracket_re = re.compile(r'\[(.*)\]')
    tl = bracket_re.search(l)

    q_re = re.compile(r'\'(.*?)\'')
    toli = q_re.findall(tl.group(1))
    token_words.append(toli)

### Word2Vec ###
sentences = token_words

model = Word2Vec(sentences, window=15, min_count=5, iter=10)
print('\n\nModel Summary (cbow + n):')
print(model)
model.wv.save('co-w2v.bin')

model_sg = Word2Vec(sentences, window=15, min_count=5, sg=1, iter=10)
print('\n\nModel Summary (sg + n):')
print(model_sg)
model_sg.wv.save('co-w2v-sg.bin')

model_sghs = Word2Vec(sentences, window=15, min_count=5, sg=1, hs=1, iter=10)
print('\n\nModel Summary (sg + hs):')
print(model_sghs)
model_sghs.wv.save('co-w2v-sghs.bin')

model_hs = Word2Vec(sentences, window=15, min_count=5, hs=1, iter=10)
print('\n\nModel Summary (cbow + hs):')
print(model_hs)
model_hs.wv.save('co-w2v-hs.bin')
