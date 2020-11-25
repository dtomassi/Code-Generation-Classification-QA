import nltk
import spacy
import jsonlines
import json
import time
import io
import tokenize
import keyword
import builtins
import ast
import re
import pickle
import os
import string
import pickle
import gensim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from collections import Counter


from nltk.stem import PorterStemmer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = PorterStemmer()
    def __call__(self, articles):
        return [self.wnl.stem(t) for t in nltk.word_tokenize(articles)]



all_stopwords = list(set(gensim.parsing.preprocessing.STOPWORDS))
STOPWORDS = all_stopwords + ['python','create','use','convert','insert','drop','way','variable',\
'row','loop','check','print','generate','line','run','differ','sort','multiple','data','change','add',\
'return','remove','value',"function","array","values",'column','columns','element','number',\
'object','key','dataframe','plot','class','item','character','text',\
'image','model','window','method','format','base','index','window','read','efficient','write','iterate','split']

p = PorterStemmer()

CONALA_MINED_FILE = "../data/conala-corpus/conala-mined.jsonl"
TRAIN_FILE = "../data/conala-corpus/conala-train.json"
TEST_FILE = "../data/conala-corpus/conala-test.json"

table = str.maketrans("","",string.punctuation)

STOPWORDS = [p.stem(t) for t in STOPWORDS]


def cleanup(intent):
	intent = intent.lower().translate(table)
	tokens = nltk.word_tokenize(intent)
	final_tokens = [p.stem(t) for t in tokens]
	final_tokens = [token for token in final_tokens if token not in STOPWORDS]
	final_intent = ' '.join(final_tokens)
	return final_intent

def get_intents(recreate = False):

	if recreate or not os.path.isfile("intents_parsed.pkl"):
		intents = []
		with jsonlines.open(CONALA_MINED_FILE,'r') as f:
			all_recs = list(f)

		#mined_intents = [rec['intent'].lower().translate(table) for rec in all_recs]
		mined_intents = [cleanup(rec['intent']) for rec in tqdm(all_recs)]

		print(f"MINED intents: {len(mined_intents)}")

		with open(TRAIN_FILE,'r') as f:
			train_recs = json.load(f)

		#train_intents = [rec['intent'].lower().translate(table) for rec in train_recs]
		train_intents = [cleanup(rec['intent']) for rec in tqdm(train_recs)]

		print(f"TRAIN intents: {len(train_intents)}")
		with open(TEST_FILE,'r') as f:
			test_recs = json.load(f)

		#test_intents = [rec['intent'].lower().translate(table) for rec in test_recs]
		test_intents = [cleanup(rec['intent']) for rec in tqdm(test_recs)]

		print(f"TEST intents: {len(test_intents)}")

		intents = mined_intents + train_intents + test_intents
		print(f"Combining all 3 together, we obtain {len(intents)} many intents")

	
		with open("intents_parsed.pkl",'wb') as f:
			pickle.dump(intents,f)

	else:
		with open("intents_parsed.pkl",'rb') as f:
			intents = pickle.load(f)

	return intents

#remove stopwords
#stemmatize every word

def vocab_stats(corpus):
	vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1)
	X = vectorizer.fit_transform(corpus)
	vocab = vectorizer.get_feature_names()
	counts = X.sum(axis=0).A1
	freq_distribution = Counter(dict(zip(vocab, counts)))
	return vocab,freq_distribution

def plot_distr(vocab):
	labels,sizes = zip(*vocab)
	fig1, ax1 = plt.subplots()
	ax1.pie(list(sizes), labels=list(labels), autopct='%1.1f%%',shadow=True, startangle=90)
	ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	plt.show()

def main():
	
	start  = time.time()
	intents = get_intents()
	
	print(intents[:20])
	
	vocab,freq_distribution = vocab_stats(intents[:])
	print(freq_distribution.most_common(50))
	plot_distr(freq_distribution.most_common(20))

	print(f"Time taken: {(time.time() - start)/60.00:.3f} minutes")

if __name__ == '__main__':
	main()


