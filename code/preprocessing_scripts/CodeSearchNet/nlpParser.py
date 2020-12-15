import json
import time
import io
import tokenize
import keyword
import builtins
import ast
import re
import pickle
import string
import os

SPECIAL_FILES = ['mysql.txt','subprocess.txt','urllib.txt']
def is_keyword(name):
	return name in keyword.kwlist

def is_builtin(name):
	return name in builtins.__dict__
def isFloat(string):
	try:
		float(string)
		return True
	except ValueError:
		return False
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

			#print(new_methods)
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

	
	api_methods = list(set(api_methods))

	return api_names,api_methods

FILE = "../data/combined-dataset.json"
API_Folder = 'API Method txt library/'

#check a token, if the token is one of the APIs (the filenames from the API_FOLDER), 
#then save that API name and then just observe the file associated with the API
api_names,api_methods = getApiCalls()

def is_api_call(token):
	return (token in api_names) or (token in api_methods)

def parsingVocab():
	vocabList = []
	print("Started Reading JSON file which contains multiple JSON document")
	stopWord=["[","]",",","{","}","(",")",":","=","==",".","0","1","2","3","4","5","6","7","8","9","10","+=","!=","<",">","<=",">=","-","+","0.6","UTC","//","_","^=","<<","/=","/","&=","~","@",">>"]
	stopWordVocab=["{","}","0","1","2","3","4","5","6","7","8","9","10","-","0.6","UTC","//","_","^=","<<","/","&=","~","@",">>"]
	with open('python_train_13.json') as f:
		for jsonObj in f:
			vocabList.append(json.loads(jsonObj))

	print("Printing each JSON Decoded Object")
	finalVarList=[]
	finalVocabList=[]
	finalVocabList2=[]
	for codeobject in vocabList:
		for i in codeobject["code_tokens"]:
			
			if i not in stopWordVocab and "#" not in i and " " not in i and "%" not in i and "\\n" not in i and "*" not in i and not i.startswith("\"") and not i.startswith("\'"):
				finalVocabList.append(i)
			if not is_builtin(i) and not is_keyword(i) and i not in stopWord and "#" not in i and " " not in i and "%" not in i and "\\n" not in i and "*" not in i:
				if not(i.isdigit()) and not isFloat(i) and not i.startswith("\"") and not i.startswith("\'") and not is_api_call(i):
					finalVarList.append(i)

		for item in finalVarList:
			if item in finalVocabList and item != '<VAR_NAME>':
				finalVocabList[finalVocabList.index(item)] = '<VAR_NAME>'
				finalVocabList2.append(finalVocabList)

		finalVocabList=[]
		finalVarList=[]
	with open("codeSearchNetVocab13.txt","w") as filehandle:
		filehandle.writelines("%s\n" % place for place in finalVocabList2)

	return finalVocabList2

def main():
	snippetList=parsingVocab()
	print("Finish Vocab")

if __name__ == '__main__':
	main()