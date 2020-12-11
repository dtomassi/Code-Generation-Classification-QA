import re
import os
from collections import Counter
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer

token_words = []
vocab = []
tfile = open('../data/fresh-combined-datasetVocab.txt','r')
L = list(tfile)
print("Total samples: ",len(L))
for l in L[:]:
    bracket_re = re.compile(r'\[(.*)\]')
    tl = bracket_re.search(l)

    q_re = re.compile(r'\'(.*?)\'')
    toli = q_re.findall(tl.group(1))
    if '' in toli:
    	toli.remove('')

    token_words.append(' '.join(toli))
    vocab += toli

counter = dict(Counter(vocab))
MIN_COUNT = 5
keys = [key for key in counter.keys() if counter[key] <= MIN_COUNT]
t = Tokenizer(lower=False,filters='',split = ' ',char_level=False,)
t.fit_on_texts(token_words)
max_words = len(t.word_index) + 1
print("Total tokens as per Tokenizer: ",max_words)
print("Total tokens: ",len(counter.keys()))
print("Number of deleted tokens: ",len(keys))
print("Final number of tokens left out: ",len(list(set(vocab))) - len(keys))