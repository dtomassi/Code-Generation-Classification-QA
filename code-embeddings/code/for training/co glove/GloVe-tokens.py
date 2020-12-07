import pandas as pd
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

### Write to Text file ###

f = open('GloVeText-co.txt', 'w')

for snippet in token_words:
    for token in snippet:
        if token != '':
            intoken = token + ' '
            f.write(intoken)

f.close()
