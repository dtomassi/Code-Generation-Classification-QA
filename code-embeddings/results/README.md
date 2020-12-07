# Results
This folder contains the results of the embeddings.

## Frequency Count
The `freq-count-bargraph` folder contains four bar graphs of the ten most frequent tokens.

- `FreqCount-combined-glove.png`: GloVe embeddings of the combined dataset
- `FreqCount-combined-w2v.png`: Word2Vec embeddings of the combined dataset
- `FreqCount-csn-glove.png`: GloVe embeddings of the CodeSearchNet + combined datasets
- `FreqCount-csn-glove.png`: Word2Vec embeddings of the CodeSearchNet + combined datasets

## PCA Graphs
The `pca-graphs` folder contains eight PCA visualizations of the embeddings, two per embeddings/corpus combination.

- `All`: This folder contains the visualization of all the tokens within each embeddings/corpus.
  - `PCAall-combined-glove.png`: GloVe embeddings of the combined dataset
  - `PCAall-combined-w2v.png`: Word2Vec embeddings of the combined dataset
  - `PCAall-csn-glove.png`: GloVe embeddings of the CodeSearchNet + combined datasets
  - `PCAall-csn-w2v.png`: Word2Vec embeddings of the CodeSearchNet + combined datasets
- `Top10`: This folder contains the visualization of the 10 most frequently appearing tokens within each embeddings/corpus.
  - `PCAfreq-combined-glove.png`: GloVe embeddings of the combined dataset
  - `PCAfreq-combined-w2v.png`: Word2Vec embeddings of the combined dataset
  - `PCAfreq-csn-glove.png`: GloVe embeddings of the CodeSearchNet + combined datasets
  - `PCAfreq-csn-w2v.png`: Word2Vec embeddings of the CodeSearchNet + combined datasets  

## Vocabulary Results
The text file `vocab-results.txt` contains information about the vocabulary of each dataset, including:

1. The number of tokens
2. 10 most frequent tokens along with the number of times they appear

## CodeSearchNet + Combined Datasets Word2Vec Training Results
The text file `w2v-csn-training-results.txt` contains insight into the training process for the CodeSearchNet + combined dataset for Word2Vec. Specifically, the file includes a snapshot of the model each time a new text file was processed (e.g. `codeSearchNetVocab0.txt`. `codeSearchNetVocab1.txt`, etc.). This snapshot includes the following:

1. Model Summary: This notably shows the vocabulary count after training the model on that particular file
2. Similarity tests: This contains similarity tests run on the model after processing a particular file
