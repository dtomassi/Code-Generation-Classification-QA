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
import itertools
import pandas as pd
from tqdm import tqdm

CONSTANT_VAR_TOKEN = "<VAR_NAME>"
SPECIAL_FILES = ['mysql.txt','subprocess.txt','urllib.txt']
FILE = "../data/combined-dataset.json"
API_Folder = 'API Method txt library/'

#check if it is a keyword in python
def is_keyword(name):
    return name in keyword.kwlist

#check if it is a builtin function/variable
def is_builtin(name):
    return name in builtins.__dict__

#this returns which APIs and the API methods associated with each file
def getApiCalls():
	api_names = ['pandas','numpy','matplotlib','pyplot']
	for names in os.listdir(API_Folder):
		api_names.append(names[:-4])

	api_methods = []
	#go through each file in the API folder
	for filename in os.listdir(API_Folder):
		#if it belongs to these files which have calls such as connector.connect().cursor().fetchall()
		if filename in SPECIAL_FILES:
			with open(API_Folder + filename) as f:
				methods = list(f) #all lines of a file f
				methods = [m.strip('\n') for m in methods] #remove the newlines for every line
				new_methods = []
				#every line 'm'
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
						#in case of a regular method call
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
					#regular method call
					if '(' in m:
						new_methods.append(m[:m.index('(')])

					else:
						new_methods.append(m)
						 
				#ensure that just the name of the method is captured: for example: loads() in the file -> loads

			api_methods += new_methods

	
	api_methods = list(set(api_methods))
	api_methods.remove('') #remove the empty call (occurs due to some faulty reason)
	return api_names,api_methods


#check a token, if the token is one of the APIs (the filenames from the API_FOLDER), 
#then save that API name and then just observe the file associated with the API
api_names,api_methods = getApiCalls()

#if it is an API name or it is an API methods
def is_api_call(token):
	return (token in api_names) or (token in api_methods)

#method to parse all the code snippet
def parse_code(snippet,to_print = False):
	

	all_names = []
	all_tokens = []
	if to_print:
		print("\nCODE\n")
		print(snippet)
		print("\nPARSED OUTPUT\n")

	try:
		#break every snippet into tokens
		for token in tokenize.generate_tokens(io.StringIO(snippet).readline):
			ttype, token_val, start, end, line = token
			#if there is a newline or an empty string
			if token_val == '\n' or token_val == '':
				continue
			all_tokens.append(token_val) #add in all the tokens
			if to_print:
				print(tokenize.tok_name[ttype],'\t',token_val)
			
			#whenever there is a 'NAME' type encountered
			if ttype == tokenize.NAME:
				#if neither builtin nor keyword nor API call nor API
				if not is_builtin(token_val) and not is_keyword(token_val) and not is_api_call(token_val):
					all_names.append(token_val) #add the token into the variable name list
					all_tokens.pop() #remove the variable name as it as from the list of tokens
					all_tokens.append(CONSTANT_VAR_TOKEN) #Add in the constant token for variable name
			
	except:
		return [],[],True #return empty lists for faulty snippets which cannot be parsed
	
	if to_print:
		print("All variable/function names:\n",all_names)
	
	
	return all_tokens,all_names,False



def get_snippets():
	#parse the combined-dataset.json files
	snippets = []
	df_json = pd.read_json(FILE) #read the json into a dataframe
	df_json = df_json.loc[:,['question_id','intent','snippet','class']] #get the important columns
	print(f"Overall, we obtain {len(df_json)} many code snippets")
	return df_json

#print the vocabulary from all tokens and all the variable names
def vocab_stats(all_tokens,all_name_vars):
	total_tokens = list(set(itertools.chain(*all_tokens)))
	total_names =  list(set(itertools.chain(*all_name_vars)))
	print(f"{len(total_tokens)} many tokens overall")
	print(f"{len(total_names)} many variable/function names")


#this runs through all the snippets
def parse_all_snippets(df_json):
	all_tokens = list()
	all_name_vars = list()
	df_final = df_json.copy() #the final dataframe
	num_faulty = 0 #count all the faulty snippets
	for i,row in tqdm(df_json.iterrows()):
		snip = row['snippet'] #get the snippet
		tokens,names,faulty = parse_code(snip,to_print=False) #get the list of parsed and normalized tokens
		if faulty:
			df_final = df_final.drop(i) #remove the faulty example
			num_faulty += 1
			continue

		all_tokens.append(tokens)
		all_name_vars.append(names)

	df_final['clean_snippet'] = all_tokens #add a seperate column of cleaned snippets
	print(f"{num_faulty} many faulty records")
	vocab_stats(all_tokens,all_name_vars) #print all the vocabulary stats
	return all_tokens,all_name_vars,df_final

#save the dataframe into the final combined dataset file
def save_to_json(df_final,filename = "../data/final-combined-dataset.json"):
	with open(filename,'w') as data_file:
		all_recs = []
		for i,row in df_final.iterrows():
			data = {"snippet": row['snippet'],\
			"intent": row['intent'],\
			"question_id": row['question_id'],\
			"class": row['class'],\
			"clean_snippet": row['clean_snippet']
			}
			all_recs.append(data)

		json.dump(all_recs, data_file, indent=4)

	with open(filename,'r') as data_file:
		json_recs = json.load(data_file)
		print(f"{len(json_recs)} many records after removing the faulty records")

#create the final list of lists code for this file
def create_list_of_lists(all_tokens,filename = 'combined-datasetVocab'):
	with open(f"{filename}.txt","w") as filehandle:
		filehandle.writelines("%s\n" % place for place in all_tokens)

def main():
	start  = time.time()
	df_json = get_snippets()
	all_tokens,all_name_vars,df_final= parse_all_snippets(df_json.loc[:,:])
	save_to_json(df_final)
	create_list_of_lists(all_tokens)
	print(f"Time taken: {(time.time() - start)/60.00:.3f} minutes")

if __name__ == '__main__':
	main()
