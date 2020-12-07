# Files Used to Obtain Results
## Files Included
### Models
- `combined-w2v-model.bin`: Word2Vec model for the combined dataset
- `csn-w2v.model`: Word2Vec model for the CodeSearchNet + combined datasets
- `combined-glove-vectors.txt`: GloVe vectors for the combined dataset
  - `combined-glove-vectors-as-w2v.txt`: Glove vectors as Word2Vec vectors for the combined dataset
- `csn-glove-vectors.txt`: GloVe vectors for the CodeSearchNet + combined datasets
  - `csn-glove-vectors-as-w2v.txt`: Glove vectors as Word2Vec vectors for the CodeSearchNet + combined datasets

### Vocab List (Provided by GloVe)
- `combined-glove-vocab.txt`: Text file that includes the vocab and vocab count for the combined dataset
- `csn-glove-vocab.txt`: Text file that includes the vocab and vocab count for the CodeSearchNet + combined datasets

### Dataset Analysis
`dataset-anaysis.py`: Created three files to analyze the resulting embeddings:

1. Writes vocab count to the text file "vocab-count.txt"
2. Creates a bar graph of the 10 most frequent tokens
3. Creates 3 Possible PCA graphs: random tokens, all tokens, and most frequent tokens

See the [results file](https://github.com/Sairamvinay/Code-Generation-Classification-QA/tree/Embeddings/code-embeddings/results) for more information.

### `for training` folder
Contains the python files used for training the models. Within them includes:
- `co glove`: Used to obtain GloVe embeddings for the combined dataset
  - `GloVe-token-format-combined.py`: Created a text file of tokens to feed into GloVe
  - `gw-co.py`: Contains similarity tests for GloVe embeddings
- `co w2v`: Used to obtain the Word2Vec embeddings for the combined dataset
  - `w2v-combo.py`: Used to create the Word2Vec model for the combined dataset
- `csn glove`: Used to obtain the GloVe embeddings for the CodeSearchNet + combined datasets
  - `GloVe-token-format-csn.py`: Created a text file of tokens to feed into GloVe, taking one corpus file at a time (of 15 total)
  - `gw-csn.py`: Contains similarity tests for GloVe embeddings
- `csn-w2v`: Used to obtain the Word2Vec embeddings for the CodeSearchNet + combined datasets
  - `w2v-train_first.py`: Used to create the Word2Vec model, training on just combined dataset. To train the model further on the CodeSearchNet corpus, `w2v-train.py` was used
  - `w2v-train.py`: Used to further train the model on the CodeSearchNet corpus one file at a time (out of 14 total)
