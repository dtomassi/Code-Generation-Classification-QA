# Creating Embeddings From .json Files
## Word2Vec
In the terminal, call the python file `w2v-all.py` with your json file as an argument.

`$ python3 w2v-all.py dataset.json`

This will tokenize the dataset in the json file, convert these tokens to vectors (word embeddings), using Word2Vec's algorithm and output the following:
- **Model summary**: This includes the vocab count, the size of the vectors, and alpha.
- **Similarity tests**: These will output the similarity score between two input tokens. The default is currently between 'print' and 'in', 'for' and 'in', and 'if' and 'else'. 
To test more tokens, edit the code (see below).
- **Tokens most similar to input**: This will output the 10 most similar tokens to the input token. The default is currently 'for'. To test more tokens, edit the code (see below).

This file can also add these results to the text file `w2v-results.txt` and show a graph of 25 random tokens based on similarity. To do this, uncomment the respective sections.


### Similarity tests
To view the comparison between two tokens, use the `saveResults()` method, which take the arguments of two tokens (strings) and the results list. For example, to see the similarity
score between the tokens 'import' and 'numpy', edit the code to include the following:

`saveResults('import','numpy',results)`

### Tokens most similar to input
To view the top 10 most similar tokens to a particular token, use the `saveResults1Word()` method, which takes the arguments of one token (string) and the results list. For example,
to see the 10 most similar tokens to the token 'int', edit the code to include the following:

`saveResults('int',results)`

## FastText
In the terminal, call the python file 'fasttext-any.py` with your json file as an argument.

`$ python3 fasttext-any.py dataset.json`

Everything else is essentially the same as Word2Vec.

## GloVe
To create and use GloVe embeddings, complete the following three steps:

### Part 1: Save Tokenize Dataset to Textfile
The first step is to use `GloVe-corpus-all.py` to tokenize your .json Dataset.

`$ python3 GloVe-corpus-all.py dataset.json`

The output will be a text file called `All-GloVeText.txt` containing the tokens of the input dataset.

### Part 2: Extract GloVe vectors
To obtain the GloVe word embeddings, move `All-GloVeText.txt` is within the `GloVe` folder. Call the `all.bash` file from terminal.

`./all.sh`

The output will include two text files:
- `all-vectors.txt`: This contains your tokens and vector embeddings for those tokens.
- `all-vocab.txt`: This contains your vocabulary (i.e. a list of tokens).

You will need the `all-vectors.txt` for the following step.

### Part 3: Using GloVe Embeddings by Converting to the Word2Vec Format
In the terminal, call the python file `GloVe-to-w2v-all.py` with `all-vectors.txt` as an argument, making sure that file has been moved to the same folder as the python file.

`$ python3 GloVe-to-all.py all-vectors.txt`

### Final GloVe Notes
Much of the preprossesing explained above has been completed and necessary files are included in the 

Following that, these vectors will be converted to word2vec embeddings and will allow the use of the same functions presented in Word2Vec and FastText.
