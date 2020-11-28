"""
Before you use:

Print tokenized vectors to text file (use GloVe-corpus-all.py), then edit ./demo
glove file in glove folder so that CORPUS=your_textfile.txt. Use the vector
textfile output as your argument for this program.
"""

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import sys
import os

vectors = sys.argv[1]

# glove_f = datapath(vectors)
glove_f = os.getcwd() + '/' + vectors
tmp_f = get_tmpfile("glove_w2v.txt")

# print(glove_f)

_ = glove2word2vec(glove_f, tmp_f)

model = KeyedVectors.load_word2vec_format(tmp_f)

results = [] # To save results

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
"""
f = open("gloveW2V-results.txt", "a")
f.write("File: " + vectors + "\n\n")
for r in results:
    f.write(r + "\n")

f.close()
"""
