# Code Snippets Embeddings

In this directory is the embeddings models, found in the folder `final_embeddings_vectors`, and the scripts used to train these models, found in the folder `training scripts`.

## Final Embeddings Vector
### Models
- `current-w2v-model-mincount0.bin`: Word2Vec model trained on the current dataset.
- `current-glove-vectors-mincount0.txt`: GloVe vectors trained on the current dataset.
- `csn-w2v.model`: Word2Vec model for the CodeSearchNet + combined datasets.
- `csn-glove-vectors-mincount0.txt`: GloVe vectors trained on the CodeSearchNet + current dataset.

### Training Scripts
Contains the python files used for training the models. Within them includes:
- `current glove`: These scripts were used to obtain GloVe embeddings for the current dataset.
  - `extractvocabtokens_Glove_current.py`: This created a text file of tokens to feed into GloVe (see `glove.sh`).
  - `inspectGlove_current.py`: Contains similarity tests for GloVe embeddings.
- `current w2v`: This script was used to obtain the Word2Vec embeddings for the current dataset.
  - `trainW2V_current.py`: Used to create the Word2Vec model for the current dataset.
- `csn glove`: These scripts were used to obtain the GloVe embeddings for the CodeSearchNet + current datasets.
  - `extractvocabtokens_Glove_csn.py`: This created a text file of tokens to feed into GloVe, taking one corpus text file at a time (of 15 total).
  - `inspectGlove_csn.py`: This contains similarity tests for GloVe embeddings.
- `csn w2v`: These scripts were used to obtain the Word2Vec embeddings for the CodeSearchNet + current datasets.
  - `trainW2V_csn_onlycurrent.py`: Used to create the Word2Vec model, training on just current dataset. To train the model further on the CodeSearchNet corpus, `trainW2V_csn_eachcsn.py` was used (see below).
  - `trainW2V_csn_eachcsn.py`: Used to further train the model on the CodeSearchNet corpus one file at a time (out of 14 total).
  - `glove.sh`: This shell file was used to create the GloVe embeddings, specifically cloning the GloVe repository, found [here](https://github.com/stanfordnlp/GloVe).

### Similarity Tests
The python script `sim_test_generate.py` in this folder was used to create similarity tests to analyze the embedding models after training. The results of these tests are in the `vocabulary` folder [here](https://github.com/Sairamvinay/Code-Generation-Classification-QA/tree/main/data/vocabulary).

## Results
This folder contains the results from analyzing the embeddings.

### Frequency Count
The `freq-count-bargraph` folder contains four bar graphs of the ten most frequent tokens.

- `FreqCount-combined-glove.png`: GloVe embeddings of the current dataset.
- `FreqCount-combined-w2v.png`: Word2Vec embeddings of the current dataset.
- `FreqCount-csn-glove.png`: GloVe embeddings of the CodeSearchNet + current datasets.
- `FreqCount-csn-glove.png`: Word2Vec embeddings of the CodeSearchNet + current datasets.

### PCA Graphs
The `pca-graphs` folder contains eight PCA visualizations of the embeddings, two per embeddings/corpus combination.

- `All`: This folder contains the visualization of all the tokens within each embeddings/corpus.
  - `PCAall-combined-glove.png`: GloVe embeddings of the current dataset.
  - `PCAall-combined-w2v.png`: Word2Vec embeddings of the current dataset.
  - `PCAall-csn-glove.png`: GloVe embeddings of the CodeSearchNet + current datasets.
  - `PCAall-csn-w2v.png`: Word2Vec embeddings of the CodeSearchNet + current datasets.
- `Top10`: This folder contains the visualization of the 10 most frequently appearing tokens within each embeddings/corpus.
  - `PCAfreq-combined-glove.png`: GloVe embeddings of the current dataset.
  - `PCAfreq-combined-w2v.png`: Word2Vec embeddings of the current dataset.
  - `PCAfreq-csn-glove.png`: GloVe embeddings of the CodeSearchNet + current datasets.
  - `PCAfreq-csn-w2v.png`: Word2Vec embeddings of the CodeSearchNet + current datasets.  

