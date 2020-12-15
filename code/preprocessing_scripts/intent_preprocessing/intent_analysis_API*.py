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
'row','loop','check','print','line','run','differ','sort','multiple','data','change','add',\
'return','remove','value',"function","array","values",'column','columns','element','number',\
'object','key','dataframe','plot','class','item','character','text',\
'image','model','window','method','format','base','index','window','read','efficient','write','iterate','split','match',\
'specific','extract','expression','module','replace','output','field','code','word','generate','parse','query']

REPLACE_WITH = {'command':'sys/os/subprocess','dict':'dictionari','url':'urllib','regular':'regex','re':'regex','argument':'argparse/subprocess/sys'\
,'select':'SQL'}
p = PorterStemmer()

#CONALA_MINED_FILE = "../data/conala-corpus/conala-mined.jsonl"
#TRAIN_FILE = "../data/conala-corpus/conala-train.json"
#TEST_FILE = "../data/conala-corpus/conala-test.json"

FILE = "../data/combined-dataset.json"

table = str.maketrans("","",string.punctuation)

STOPWORDS = [p.stem(t) for t in STOPWORDS]


def cleanup(intent):
	intent = intent.lower().translate(table)
	tokens = nltk.word_tokenize(intent)
	final_tokens = [p.stem(t) for t in tokens]
	final_tokens = [final_token for token,final_token in zip(tokens,final_tokens) if (final_token not in STOPWORDS)]
	final_tokens = [REPLACE_WITH.get(final_token,final_token) for final_token in final_tokens] #replace common terms
	final_intent = ' '.join(final_tokens)
	return final_intent

def get_intents(recreate = False):

	if recreate or not os.path.isfile("intents_parsed.pkl"):
		intents = []
		orig_intents = []
		with open(FILE,'r') as f:
			all_recs = json.load(f)
			orig_intents = [rec['intent'] for rec in all_recs]

		print("TOTAL INTENTS: ",len(orig_intents))
		print("Original intents: \n",orig_intents[:20])
		intents = [cleanup(intent) for intent in tqdm(orig_intents[:])]

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
	return vocab,freq_distribution,sum(counts)

def plot_distr(vocab,count):
	labels,sizes = zip(*vocab)
	total = sum(sizes)
	fig1, ax1 = plt.subplots()
	ax1.pie(list(sizes), labels=list(labels), autopct='%1.1f%%',shadow=True, startangle=90)
	ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	plt.title(f"Vocabulary size: {count}")
	plt.legend(
    loc='upper left',
    labels=['%s, %d, %1.1f%%' % (
         l, s,(float(s) / total) * 100) for l, s in zip(labels, sizes)],
    prop={'size': 11},
    bbox_to_anchor=(0.0, 1),
    bbox_transform=fig1.transFigure)
	plt.show()

def main():
	
	start  = time.time()
	intents = get_intents(recreate=  True)
	
	print("PARSED intents:\n",intents[:20])
	
	vocab,freq_distribution,total_vocab_count = vocab_stats(intents[:])
	print(freq_distribution.most_common(50))
	plot_distr(freq_distribution.most_common(20),total_vocab_count)

	print(f"Time taken: {(time.time() - start)/60.00:.3f} minutes")

if __name__ == '__main__':
	main()


