import nltk
import json
import time
import ast
import re
import os
import pandas as pd
import string
import pickle
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

FILE = "../data/pos_neg_final_dataset.json"

#need to clean up the intents a little, a short script 
#to remove the specific charaacters and the upper cases into lower cases
#remove few chars: ? . ! 
#remove useless spaces into one space if any
#add the space for these chars: \" , 
def cleanup(intent):
	intent = intent.lower()
	intent = intent.replace('?','')
	intent = intent.replace('!','')
	intent = intent.replace(':','')
	intent = intent.replace('\"','')
	intent = intent.replace("\'",'')
	intent = intent.replace("-",'')
	#intent = ' '.join(nltk.word_tokenize(intent))
	return intent


def graph_distr(distr):
	labels = distr.keys()
	counts = distr.values()
	labels_dict = {0: "negative", 1: "positive"}
	label_names = [labels_dict[label] for label in labels]
	plt.bar(label_names,counts)
	plt.xlabel("The Class type of the intent-snippet pair")
	plt.ylabel("the number of samples")
	plt.title("Frequency distribution of the intent-snippet pair")
	plt.text(0.5,43000,f"Positive:{distr[1]}, negative:{distr[0]}")
	plt.show()

def get_intents(df):

	df_json = df.copy()
	orig_intents = df['intent'].values
	labels = df['class'].values
	cleaned_intents = [cleanup(intent) for intent in tqdm(orig_intents)]
	distr = dict(Counter(labels))
	print("Original intents: \n",orig_intents[:50])
	print("The label counts: \n",dict(Counter(labels)))
	df_json['cleaned_intents'] = cleaned_intents
	print("\nCleaned intents: \n",df_json['cleaned_intents'].values[:50])
	#graph_distr(distr)

	return df_json

def save_to_json(df_final,filename = "../data/overall-pos_neg_final-dataset.json"):
	with open(filename,'w') as data_file:
		all_recs = []
		for i,row in df_final.iterrows():
			data = {"snippet": row['snippet'],\
			"intent": row['intent'],\
			"question_id": row['question_id'],\
			"class": row['class'],\
			"clean_snippet": row['clean_snippet'],\
			'cleaned_intent':row['cleaned_intents']
			}
			all_recs.append(data)

		json.dump(all_recs, data_file, indent=4)

	with open(filename,'r') as data_file:
		json_recs = json.load(data_file)
		print(f"{len(json_recs)} many records after removing the faulty records")

def get_file():
	df_json = pd.read_json(FILE) #read the json into a dataframe
	df_json = df_json.loc[:,:] #get the important columns
	print(f"Overall, we obtain {len(df_json)} many code snippets")
	return df_json

def main():
	df = get_file()
	df_json = get_intents(df)
	save_to_json(df_json)

if __name__ == '__main__':
	main()