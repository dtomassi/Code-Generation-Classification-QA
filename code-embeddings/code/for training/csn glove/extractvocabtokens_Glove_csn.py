import re
import sys

### Load in Corpus Text File ###
token_words = []

filename = sys.argv[1]
tfile = open(filename, 'r')
print('Loading ' + str(filename) + '...')

for l in tfile:
    bracket_re = re.compile(r'\[(.*)\]')
    tl = bracket_re.search(l)

    q_re = re.compile(r'\'(.*?)\'')
    try:
        toli = q_re.findall(tl.group(1))
    except AttributeError:
        pass
    token_words.append(toli)

print(str(filename) + ' loaded!\n')

print('Number encoded docs: {:,}'.format(len(token_words)))
token_docs = [
    ' '.join(array) for array in token_words
]
# print("token_words[1:3]:")
# print(token_words[1:3])

### Write to Text file ###

f = open('GloVeText-csn.txt', 'a')

for snippet in token_words:
    for token in snippet:
        if token != '':
            intoken = token + ' '
            f.write(intoken)

f.close()
