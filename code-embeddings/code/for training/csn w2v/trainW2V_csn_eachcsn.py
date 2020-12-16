from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot
import re
import sys

### Load in Model ###
model = Word2Vec.load('w2v-csn.model')

### Load in Corpus Text File ###
token_words = []
results = []

filename = sys.argv[1]
tfile = open(filename, 'r')
print('Loading ' + str(filename) + '...')

for l in tfile:
    bracket_re = re.compile(r'\[(.*)\]')
    tl = bracket_re.search(l)

    q_re = re.compile(r'\'(.*?)\'')
    try:
        toli = q_re.findall(tl.group(1))
    except AttributeError:
        pass
    token_words.append(toli)

print(str(filename) + ' loaded!\n')


### Word2Vec ###
sentences = token_words
te = len(token_words)
model.build_vocab(token_words, update=True)
model.train(sentences, total_examples=te, epochs=model.epochs)

print('\n\nModel Summary:')
print(model)
model.save('co-w2v.bin')

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
f.write("File Added: " + filename + "\n\n")
for r in results:
    f.write(r + "\n")
f.write('\n\n')

f.close()
