# Vocabulary

The files within this directory are all vocabulary related results from training the embedding models.

## Vocabulary Results

The following text files are the result of similarity tests done during or after training the embedding models. These include a similarity test involving two tokens and a similarity test that lists the 10 most similar tokens to a particular token.

- `glove-csn-training-result.txt`: This contains the similarity test result of the CodeSearchNet + current datasets with GloVe. This was obtained using the `sim_result_generate.py` script, found [here](https://github.com/Sairamvinay/Code-Generation-Classification-QA/tree/main/code/Embeddings/code_snippet_embedding).
- `glove-current-training-result.txt`: This contains the similarity test result of the current datasets with GloVe. This was obtained using the `sim_result_generate.py` script, found [here](https://github.com/Sairamvinay/Code-Generation-Classification-QA/tree/main/code/Embeddings/code_snippet_embedding).
- `w2v-csn-training-results.txt`: This contains the similarity test result of the CodeSearchNet + current datasets with Word2Vec during training. As each text file was used to train the model, a similarity test was conducted to see how the model shifted as the more and more data was used.
- `w2v-current-training-result.txt`: This contains the similarity test result of the current datasets with Word2Vec. This was obtained using the `sim_result_generate.py` script, found [here](https://github.com/Sairamvinay/Code-Generation-Classification-QA/tree/main/code/Embeddings/code_snippet_embedding).

## GloVe Vocab Output

GloVe's training results in the output of the vectors as well as a vocabulary text file that inclues the list of tokens within the vocabulary of the trained model and the count.

- `csn-glove-vocab-mincount0.txt`: This is the vocabulary for the CodeSearchNet + current datasets.
- `w2v-current-training-result.txt`: This is the vocabulary for the current dataset.