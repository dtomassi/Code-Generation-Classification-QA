import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tokenize import generate_tokens
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from keras.preprocessing.text import Tokenizer
from sklearn.decomposition import PCA
from matplotlib import pyplot
import sys

dataset = sys.argv[1]
DSname = dataset.split('.')
filename = DSname[0]

#### Open File ###
df = pd.read_json(dataset)
df = df.iloc[0:]

NUM_WORDS=20000

def get_doc(df):
    _doc = []
    for i in df.index:
        code = df.at[i, 'snippet']
        _doc.append(code)
    return _doc

doc = get_doc(df)

### Tokenize ###
token_ids = []
token_words = []

i = 0
while i < len(doc):
    docs = doc[i]
    try:
        tokens = [(t[0],t[1]) for t in list(generate_tokens(StringIO(docs).readline))]
        # Token ID
        token_ids.append(
            [token[0] for token in tokens]
        )
        # Token
        token_words.append(
            [token[1] for token in tokens]
        )
    except:
        pass
    i += 1

print('Number encoded docs: {:,}'.format(len(token_words)))
token_docs = [
    ' '.join(array) for array in token_words
]

results = [] # To save results

### Word2Vec ###
sentences = token_words
model = Word2Vec(sentences, min_count=5)
# Add to results
results.append("-------Model Summary--------")
results.append(str(model))

words = list(model.wv.vocab)

### Check how similar two input tokens are to each other ###
results.append('\n-------Similarity Tests-------')

# Prints similarity between input tokens and saves to results
def saveResults(w1,w2,r):
    r.append("Similarity between '" + w1 + "' and '" + w2 + "': " + str(model.wv.similarity(w1, w2)))

saveResults('print','in',results)
saveResults('for','in',results)
saveResults('if','else',results)

### Check most similar tokens to input ###
results.append('\n-------Tokens Most Similar to Input-------')

# Prints 10 most similar words to input and save to results
def saveResults1Word(word,r):
    r.append("Most Similar words to the token '" + word + "': ")
    top_sim = model.wv.most_similar(word)

    r.append("--Word-- \t\t\t--Similarity Score--")
    for s in top_sim:
        r.append("'" + s[0] + "'\t\t\t\t" + str(s[1]))


saveResults1Word('for',results)

### Print Results ###
print("Results:")
for r in results:
    print(r)

### Write Results to Text File ###
# Uncomment to write new results to file
"""
f = open("w2v-results.txt", "a")
f.write("File: " + filename + "\n\n")
for r in results:
    f.write(r + "\n")

f.close()
"""
### Plot Data ###
# Uncomment to view PCA graph of 25 random tokens. Change the number of tokens by
# editing the variable NumPts.
"""
x = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(x)

NumPts = 25
pyplot.scatter(result[:NumPts, 0], result[:NumPts, 1])
words = list(model.wv.vocab)[:NumPts]
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i,1]))
pyplot.show()
"""
