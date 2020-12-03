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
from tqdm import tqdm

SPECIAL_FILES = ['mysql.txt','subprocess.txt','urllib.txt']
def is_keyword(name):
    return name in keyword.kwlist

def is_builtin(name):
    return name in builtins.__dict__

def getApiCalls():
	api_names = ['pandas','numpy','matplotlib','pyplot']
	for names in os.listdir(API_Folder):
		api_names.append(names[:-4])

	api_methods = []
	for filename in os.listdir(API_Folder):
		if filename in SPECIAL_FILES:
			with open(API_Folder + filename) as f:
				methods = list(f)
				methods = [m.strip('\n') for m in methods]
				new_methods = []
				for m in methods:
					if '.' in m:
						splitted_methods = m.split('.') #in case of method cases like: connector.connect().cursor().fetchall() ; get: connector, connect, cursor, fetchall seperately 
						for sub_m in splitted_methods:
							#in case of the connect() case above
							if '(' in sub_m:
								new_methods.append(sub_m[:sub_m.index('(')])
							#in case of the connector case above
							else:
								new_methods.append(sub_m)

					else:
						if '(' in m:
							new_methods.append(m[:m.index('(')])
						else:
							new_methods.append(m)

			
			api_methods += new_methods	


		else:
			with open(API_Folder + filename) as f:

				methods = list(f)
				methods = [m.strip('\n') for m in methods]
				new_methods = []
				for m in methods:
					if '(' in m:
						new_methods.append(m[:m.index('(')])

					else:
						new_methods.append(m)
						 
				#ensure that just the name of the method is captured: for example: loads() in the file -> loads

			api_methods += new_methods

	print("API NAMES")
	print(api_names)
	print('\n'*4)

	print("METHODS")
	api_methods = list(set(api_methods))
	api_methods.remove('')
	print("NUM API METHODS:",len(api_methods))
	print(api_methods[:20])
	return api_names,api_methods

FILE = "../data/combined-dataset.json"
API_Folder = 'API Method txt library/'

#check a token, if the token is one of the APIs (the filenames from the API_FOLDER), 
#then save that API name and then just observe the file associated with the API
api_names,api_methods = getApiCalls()

def is_api_call(token):
	return (token in api_names) or (token in api_methods)

def parse_code(snippet,to_print = False):
	

	all_names = []
	all_tokens = []
	
	if to_print:
		print("\nCODE\n")
		print(snippet)
		print("\nPARSED OUTPUT\n")

	try:
		for token in tokenize.generate_tokens(io.StringIO(snippet).readline):
			ttype, token_val, start, end, line = token
			all_tokens.append(token_val)
			if to_print:
				print(tokenize.tok_name[ttype],'\t',token_val)
			if ttype == tokenize.NAME:
				if not is_builtin(token_val) and not is_keyword(token_val) and not is_api_call(token_val):
					all_names.append(token_val)
	except:
		print(f"Faulty snippet,{snippet}")
		return [],[]
	all_names = list(set(all_names))
	if to_print:
		print("All variable/function names:\n",all_names)
	
	
	return list(set(all_tokens)),all_names



def get_snippets():
	#parse the Conala mined files
	snippets = []
	with open(FILE,'r') as f:
		all_recs = json.load(f)
		snippets = [rec['snippet'] for rec in all_recs]

	print(f"Combining all 3 together, we obtain {len(snippets)} many code snippets")


	return snippets

def vocab_stats(all_tokens,all_name_vars):
	print(f"{len(all_tokens)} many tokens overall")
	print(f"{len(all_name_vars)} many variable/function names")


def parse_all_snippets(snippets):
	all_tokens = list()
	all_name_vars = list() #need to take care of this step
	for i in tqdm(range(len(snippets))):
		snip = snippets[i]
		print(f"\n{i+1}.")
		tokens,names = parse_code(snip,to_print=True)
		all_tokens += tokens
		all_tokens = list(set(all_tokens))

		all_name_vars += names
		all_name_vars = list(set(all_name_vars))

		if (i%100000 == 0 and i != 0):
			with open(f"vocabulary/Vocab_{i}.pkl","wb") as file:
				pickle.dump(all_tokens,file)

	vocab_stats(all_tokens,all_name_vars)
	#with open(f"vocabulary/Vocab_all.pkl","wb") as file:
	#	pickle.dump(all_tokens,file)

	return 
def main():
	start  = time.time()
	snippets = get_snippets()
	parse_all_snippets(snippets[:50])
	print(f"Time taken: {(time.time() - start)/60.00:.3f} minutes")

if __name__ == '__main__':
	main()