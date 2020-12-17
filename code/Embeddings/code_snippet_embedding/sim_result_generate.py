# This script loads in embeddings models and uses the gensim functions to analyze the embeddings through
# similarity tests. The first test compares two tokens based on their similarities while the second test
# outputs a token's 10 most similar tokens.

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from pathlib import Path


### Load in Models ###

# Current Dataset - w2v Model #
curr_model_w2v = KeyedVectors.load('current-w2v-model-mincount0.bin')


# Current Dataset - glove Model #
p = str(Path.cwd())
glove_f_co = datapath(p + '/current-glove-vectors-mincount0.txt')
tmp_f_co = get_tmpfile(p + '/current-glove-vectors-as-w2v.txt')

_ = glove2word2vec(glove_f_co, tmp_f_co)

curr_model_g = KeyedVectors.load_word2vec_format(tmp_f_co)


# CodeSearchNet + Current Dataset - glove Model #
glove_f_csn = datapath(p + '/csn-glove-vectors-mincount0.txt')
tmp_f_csn = get_tmpfile(p + '/csn-glove-vectors-as-w2v.txt')

_ = glove2word2vec(glove_f_csn, tmp_f_csn)

csn_model_g = KeyedVectors.load_word2vec_format(tmp_f_csn)



### Run Similarity Tests ###

def simTests(model):
	results = []

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
	results.append('\n-------Most Similar Tokens Tests-------')


	def saveResultsMostSimilar(w,r):
		r.append('Most similar tokens to \"' + w + '\":')
		s = model.wv.most_similar(w)
		n = 1

		for word, score in s:
			r.append(str(n) + '. ' + word + ': ' + str(score))
			n += 1
		print(model.wv.most_similar(w))
		results.append('\n')

	saveResultsMostSimilar('for',results)
	saveResultsMostSimilar('import',results)

	return results

### Save results in lists via running simTests method ###

r_curr_w = []
r_csn_g = []
r_curr_g = []

r_curr_w = simTests(curr_model_w2v)
r_csn_g = simTests(curr_model_g)
r_curr_g = simTests(csn_model_g)



### Write results to respected text files ###

# Current Word2Vec #
f = open("w2v-current-training-result.txt", "w")
f.write("Dataset: Current, Embeddings: Word2Vec \n\n")
for r in r_curr_w:
    f.write(r + "\n")
f.write('\n\n')

f.close()


# CSN + Current GloVe #
f = open("glove-csn-training-result.txt", "w")
f.write("Dataset: CodeSearchNet + Current, Embeddings: GloVe \n\n")
for r in r_csn_g:
    f.write(r + "\n")
f.write('\n\n')

f.close()


# Current GloVe #
f = open("glove-current-training-result.txt", "w")
f.write("Dataset: Current, Embeddings: GloVe \n\n")
for r in r_curr_g:
    f.write(r + "\n")
f.write('\n\n')

f.close()