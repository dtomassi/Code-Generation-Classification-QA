from gensim.models import Word2Vec
import re
import string
import json
import pandas as pd

def w2v_intents(filename):
	with open(filename) as f:
		json_file = json.load(filename)

	df = pd.DataFrame(json_file)
	intents = df['cleaned_intent'].apply(lambda x: x.split(' ')).values.tolist()
	model = Word2Vec(sentences = intents, size = 100, window = 4, min_count = 1, sg = 1, iter = 15)
	print(model)
	print(model.wv.vocab)
	model.wv.save('intents-w2v.model')
	return model





def main():
	model = w2v_intents("overall-pos_neg_final-dataset.json")

if __name__ == '__main__':
	main()