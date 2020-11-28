import pandas as pd
from tokenize import generate_tokens
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import sys

dataset = sys.argv[1]

#### Open File ###
df = pd.read_json(dataset)
df = df.iloc[0:]

NUM_WORDS=20000

def get_doc(df):
    _doc = []
    for i in df.index:
        code = df.at[i, 'snippet']
        _doc.append(code)
    return _doc

doc = get_doc(df)
# print('{:,}'.format(len(doc)))

### Tokenize and Preprossess for Textfile ###
token_ids = []
token_words = []

i = 0
while i < len(doc):
    docs = doc[i]
    try:
        tokens = [(t[0],t[1]) for t in list(generate_tokens(StringIO(docs).readline))]
        # Token ID
        token_ids.append(
            [token[0] for token in tokens]
        )
        # Token
        token_words.append(
            [token[1] for token in tokens]
        )
    except:
        pass
    i += 1

print('Number encoded docs: {:,}'.format(len(token_words)))
token_docs = [
    ' '.join(array) for array in token_words
]

### Write to Text file ###
f = open('All-GloVeText.txt', 'w') # Edit to change output textfile name

for snippet in token_words:
    for token in snippet:
        if token != '':
            intoken = token + ' '
            f.write(intoken)

f.close()
