import sys
import json,pickle
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


word = str(sys.argv[1])
print(word)
FILE = "../data/combined-dataset.json"
with open(FILE,'r') as f:
	all_recs = json.load(f)
	orig_intents = [rec['intent'] for rec in all_recs]

with open("intents_parsed.pkl",'rb') as f:
	cleaned_intents = pickle.load(f)

k_script = [k for k,i in enumerate(cleaned_intents) if word in i]
int_script = [orig_intents[k] for k in k_script]
vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1)
X = vectorizer.fit_transform(int_script)
vocab = vectorizer.get_feature_names()
counts = X.sum(axis=0).A1
freq_distribution = Counter(dict(zip(vocab, counts)))
print(freq_distribution)
