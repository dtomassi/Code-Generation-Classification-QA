# Code-Generation-Classification-QA

Dataset found at: https://drive.google.com/drive/u/1/folders/17EXFRp9fd3f7O3mjjsC_KJK6hLgw6WbK


## Modelling

To run the training files 

`hidden_state_model.py`: the script to run the element wise product of the hidden state and the context vector into the binary classifier

`embedding_avg_model.py`: the script to run the average of the pretrained embedding of each of the predicted code token sequence into the binary classifier

1. ```python3 hidden_state_model.py <PRETRAINED CODE EMBEDING FILENAME> "EMBEDDING FILENAME: W(ord2vec)/G(loVe)```


2. ```python3 embedding_avg_model.py <PRETRAINED CODE EMBEDING FILENAME> "EMBEDDING FILENAME: W(ord2vec)/G(loVe)```

Each of the model files are updated in the respective directories here.


## Data Generation Curation

These scripts output final dataset used for project from Conala and StaQC 

`remove_duplicates.py`: the script to filter out similar questions and low probability answers in conala-mined.jsonl. Creates curated-mined.json

`mine-train-test.ipynb`: the script to combine and deduplicate questions from curated-mined with conala-test.json and conala-train.json. It creates the conala-all.json for the combined filtered Conala dataset.

`load-StaQC.py`: the script to load and clean intent and code snippet pairs from the StaQC dataset, and create new file with them called new_staqc.json 

`overlap-between-datasets.ipynb`: the script to combine the conala and StaQC datasets together by removing duplicate questions between them.

`generate_negative_data.py`: the script to create negative code snippet samples from the combined dataset dataset. Creates pos_neg_final_dataset.json.

`dataset_intent_cleanup.py`: takes in input from data/pos_neg_final_dataset.json. Cleans punctuation from input intent questions, and outputs overall-pos_neg_final-dataset.json, the final dataset file used for the project.
