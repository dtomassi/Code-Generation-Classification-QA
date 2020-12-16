# Code-Generation-Classification-QA

Dataset found at: https://drive.google.com/drive/u/1/folders/17EXFRp9fd3f7O3mjjsC_KJK6hLgw6WbK


## Modelling

To run the training files 

`hidden_state_model.py`: the script to run the element wise product of the hidden state and the context vector into the binary classifier

`embedding_avg_model.py`: the script to run the average of the pretrained embedding of each of the predicted code token sequence into the binary classifier

1. ```python3 hidden_state_model.py <PRETRAINED CODE EMBEDING FILENAME> "EMBEDDING FILENAME: W(ord2vec)/G(loVe)```


2. ```python3 embedding_avg_model.py <PRETRAINED CODE EMBEDING FILENAME> "EMBEDDING FILENAME: W(ord2vec)/G(loVe)```

Each of the model files are updated in the respective directories here.
