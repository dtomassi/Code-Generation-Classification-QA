import pandas as pd
import json
import re
from collections import Counter


traces_filename = "../data/overall-pos_neg_final-dataset.json"
def get_clean_snippets():
	with open(traces_filename) as trace_file:
		json_traces = json.load(trace_file)

	df = pd.DataFrame(json_traces)
	clean_snippets = df['clean_snippet'].values.tolist()
	print(len(clean_snippets))
	print(clean_snippets[0])
	return clean_snippets

def process_vocab_file(clean_snippets,filename = '../data/fresh-combined-datasetVocab'):
	with open(f"{filename}.txt","w") as filehandle:
		filehandle.writelines("%s\n" % place for place in clean_snippets)

def main():
	clean_snippets = get_clean_snippets()
	process_vocab_file(clean_snippets)


if __name__ == '__main__':
	main()
