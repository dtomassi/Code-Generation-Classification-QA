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
from tqdm import tqdm

def is_keyword(name):
    return name in keyword.kwlist

def is_builtin(name):
    return name in builtins.__dict__

CONALA_MINED_FILE = "data/conala-corpus/conala-mined.jsonl"
TRAIN_FILE = "data/conala-corpus/conala-train.json"
TEST_FILE = "data/conala-corpus/conala-test.json"

class AnalysisNodeVisitor(ast.NodeVisitor):
    def visit_Import(self,node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_ImportFrom(self,node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self,node):
        print('Node type: Assign and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)
    
    def visit_BinOp(self, node):
        print('Node type: BinOp and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Expr(self, node):
        print('Node type: Expr and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Num(self,node):
        print('Node type: Num and fields: ', node._fields)

    def visit_Name(self,node):
        print('Node type: Name and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Str(self, node):
        print('Node type: Str and fields: ', node._fields)

def parse_code(snippet,to_print = False):
	

	all_names = []
	builtin_fns = []
	all_tokens = []
	lasttoken = ''
	
	if to_print:
		print(snippet)

	p = ast.parse(snippet)
	print(ast.dump(p))
	v = AnalysisNodeVisitor()
	v.visit(p)

	
	for token in tokenize.generate_tokens(io.StringIO(snippet).readline):
		ttype, token_val, start, end, line = token
		all_tokens.append(token_val)

		if to_print:
			print(tokenize.tok_name[ttype],'\t',token_val)
		if ttype == tokenize.NAME:
			if not is_builtin(token_val) and not is_keyword(token_val):
				all_names.append(token_val)

	all_names = list(set(all_names))
	if to_print:
		print("All variable/function names:\n",all_names)
	
	
	return list(set(all_tokens)),all_names



def get_snippets():
	#parse the Conala mined files
	snippets = []
	with jsonlines.open(CONALA_MINED_FILE,'r') as f:
		all_recs = list(f)


	mined_snippets = [rec["snippet"] for rec in all_recs]
	print(f"MINED snippets: {len(mined_snippets)}")

	with open(TRAIN_FILE,'r') as f:
		train_recs = json.load(f)

	train_snips = [rec["snippet"] for rec in train_recs]

	print(f"TRAIN snippets: {len(train_snips)}")
	with open(TEST_FILE,'r') as f:
		test_recs = json.load(f)

	test_snips = [rec["snippet"] for rec in test_recs]
	print(f"TEST snippets: {len(test_snips)}")

	snippets = mined_snippets + train_snips + test_snips
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
		tokens,names = parse_code(snip,to_print=False)
		all_tokens += tokens
		all_tokens = list(set(all_tokens))

		all_name_vars += names
		all_name_vars = list(set(all_name_vars))

		if (i%100000 == 0 and i != 0):
			with open(f"vocabulary/Vocab_{i}.pkl","wb") as file:
				pickle.dump(all_tokens,file)

	vocab_stats(all_tokens,all_name_vars)
	with open(f"vocabulary/Vocab_all.pkl","wb") as file:
		pickle.dump(all_tokens,file)

	return 
def main():
	start  = time.time()
	snippets = get_snippets()
	snippet = input("Enter: ")
	parse_code(snippet,to_print=True)
	print(f"Time taken: {(time.time() - start)/60.00:.3f} minutes")

if __name__ == '__main__':
	main()