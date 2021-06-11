# Code-Generation-Classification-QA

Final Report: [report.pdf](https://github.com/dtomassi/Code-Generation-Classification-QA/blob/main/report.pdf)

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

## Preprocessing_scripts

`nlpParser.py`: the script is used for CodeSearchNet dataset in order to tokenize the each code snippet and normalizing the user defined variable to filter the junk variable as <varname>. It will takes in each JSON data files from CodeSearchNet that contain codesnipppet and output the normalized version of a huge vocab list.

`code_parser.py`:  the script is used for Conala dataset in order to tokenize the each code snippet and normalizing the user defined variable to filter the junk variable as <varname>. It will read through code snippet from CodeSearchNet that contain Conala and output the normalized version of a huge vocab list.

`scraper.py` : the script will reading through 14 JSON data files from CodeSearchNet Api. For each json files, it will access each url for the json object and web scripted in order to get the import libaries and modules. Then, it will store all the libraries in a dictionary with its count as the value. Then, this is the web scraper python script for the Code Search Net in order to get the top used Api.

`snippet-vocab-check.py` : this parser takes in newest Conala Dataset and perform some cleaning on the tokenized vocal of the code snippet. Then , it will output the total tokens, token token per each tokenizer for each score snippet, number of deleted(bad) tokens and final numbers clean left out tokens.

`generate_vocab.py`: this script will read through a clean overall Conala dataset and access “clean snippets” which is the list of tokenized vocab of code snippet. It will output all the toknitized tokens to a single text file that contain all vocabulary.

`check_word_surrounding.py`: This file takes in a simple word and picks all those samples which have this word in the intent and then find the most common words that appear in these samples. in short which words are very close to each other.

`intent_analysis_API.py`: this script will do some cleaning of the intent from the Conala dataset which the intent is the question that user will ask. It will filter out based on stop words and replace common terms. It will also  provide a plot that shows distribution on the intent vocab.
