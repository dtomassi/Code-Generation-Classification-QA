from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot
import re

#### Open File ###
tfile = open('combined-datasetVocab.txt','r')

token_words = []
results = []

for l in tfile:
    bracket_re = re.compile(r'\[(.*)\]')
    tl = bracket_re.search(l)

    q_re = re.compile(r'\'(.*?)\'')
    toli = q_re.findall(tl.group(1))
    token_words.append(toli)

### Word2Vec ###
sentences = token_words

model = Word2Vec(sentences, window=15, min_count=5, sg=1, iter=10)
print('\n\nModel Summary:')
print(model)
model.save('w2v-csn.model')

## Edit results with added file + model summary
results.append("-------Model Summary--------")
results.append(str(model))

# Prints similarity between input tokens and saves to results
results.append('\n-------Similarity Tests-------')

def saveResults(w1,w2,r):
    r.append("SG: Similarity between '" + w1 + "' and '" + w2 + "': " + str(model.wv.similarity(w1, w2)))
    print('%r\t%r\t%.2f' % (w1, w2, model.wv.similarity(w1, w2)))

print('\nSimilarity Scores Between Words:')
saveResults('print','in',results)
saveResults('for','in',results)
saveResults('if','else',results)
saveResults('import','from',results)

# Check most similar tokens to input
print('\nTokens Most Similar to Input')
print(model.wv.most_similar('for'))

f = open("w2v-results.txt", "a")
f.write("File Added: combined-datasetVocab.txt\n\n")
for r in results:
    f.write(r + "\n")
f.write('\n\n')

f.close()
