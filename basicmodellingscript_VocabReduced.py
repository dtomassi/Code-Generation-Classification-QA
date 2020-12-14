# -*- coding: utf-8 -*-
"""BasicModellingScript.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y65JoXIeFpxEgh0bARmmPVF5wlO6LWCo

# **Basic setup**
"""

import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from pathlib import Path
import nltk
import json
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict,Counter

traces_filename = "overall-pos_neg_final-dataset.json"

CODE_VOCAB_SIZE = 5000
pad_length_snippet = 50

with open(traces_filename) as trace_file:
  json_traces = json.load(trace_file)

final_data = []
for datum in json_traces:
  snippet_length = len(datum['clean_snippet'])
  if snippet_length > pad_length_snippet:
    continue
  
  else:
    final_data.append(datum)

df = pd.DataFrame(final_data)

df

"""# **TRAIN-TEST-VAL Split**"""

df_train_all,df_test = train_test_split(df,test_size = 0.2,random_state = 999)
print(len(df_train_all))
print(len(df_test))

"""80-20 Train-test split
and then last 10% of train is used as val
"""

df_train = df_train_all.iloc[:-int(0.1*68128),:]
df_val = df_train_all.iloc[-int(0.1*68128):,:]
print(len(df_train))
print(len(df_val))

df_train['class'].value_counts()

df_val['class'].value_counts()

df_test['class'].value_counts()

"""# **Custom tokenizer**"""

input_decoder_snippets = df['clean_snippet'].apply(lambda snippet: ["<START>"] + snippet).values.tolist()
output_decoder_snippets = df['clean_snippet'].apply(lambda snippet: snippet + ["<END>"]).values.tolist()
print(input_decoder_snippets[:4])
print(output_decoder_snippets[:4])

vocab = []
for tokens in input_decoder_snippets + output_decoder_snippets:
  for token in tokens:
    vocab.append(token)

print(len(set(vocab)))

word_counts = dict(Counter(vocab))
wcounts = list(word_counts.items())
wcounts.sort(key=lambda x: x[1], reverse=True)
wcounts[:15]

sorted_voc = []
sorted_voc.extend(wc[0] for wc in wcounts)

word_index = dict(zip(sorted_voc, list(range(0, len(sorted_voc)))))
list(word_index.items())[:15]

len(word_index) + 1

updated_word_index = dict(list(word_index.items())[:CODE_VOCAB_SIZE])
list(updated_word_index.keys())[:10]

OOV = '<OOV>'
updated_word_index[OOV] = CODE_VOCAB_SIZE

def text2seq(snippets,w2i = updated_word_index, maxlen = CODE_VOCAB_SIZE):
  sequences = []
  for tokens in snippets:
    seq = []
    for token in tokens:
      if token in w2i:
        index = w2i[token]
        seq.append(index)
      
      else:
        seq.append(w2i[OOV])
    
    sequences.append(seq)
  
  return sequences

"""# **Tokenize the intent and the snippets**

Reference: https://stackabuse.com/python-for-nlp-neural-machine-translation-with-seq2seq-in-keras/
"""

tok_intent = Tokenizer()

input_encoder_intents = df['cleaned_intent'].values.tolist()
print(input_encoder_intents[:4])

tok_intent.fit_on_texts(input_encoder_intents)

# summarize what was learned
print(tok_intent.word_counts)
print(tok_intent.document_count)
print(tok_intent.word_index)
print(tok_intent.word_docs)

with open("tok-intent.pkl",'wb') as f:
  pickle.dump(tok_intent,f)

with open("tok-snippet.pkl",'wb') as f:
  pickle.dump(word_index,f)

input_decoder_snippets = df['clean_snippet'].apply(lambda snippet: ["<START>"] + snippet).values.tolist()
output_decoder_snippets = df['clean_snippet'].apply(lambda snippet: snippet + ["<END>"]).values.tolist()

print(input_decoder_snippets[:4])
print(output_decoder_snippets[:4])

train_intent_sequences = tok_intent.texts_to_sequences(df_train['cleaned_intent'].values.tolist())
test_intent_sequences = tok_intent.texts_to_sequences(df_test['cleaned_intent'].values.tolist())
val_intent_sequences = tok_intent.texts_to_sequences(df_val['cleaned_intent'].values.tolist())

train_input_snippet_sequences = text2seq(df_train['clean_snippet'].apply(lambda snippet: ["<START>"] + snippet).values.tolist())
test_input_snippet_sequences = text2seq(df_test['clean_snippet'].apply(lambda snippet: ["<START>"] + snippet).values.tolist())
val_input_snippet_sequences = text2seq(df_val['clean_snippet'].apply(lambda snippet: ["<START>"] + snippet).values.tolist())

train_output_snippet_sequences = text2seq(df_train['clean_snippet'].apply(lambda snippet: snippet + ["<END>"]).values.tolist())
test_output_snippet_sequences = text2seq(df_test['clean_snippet'].apply(lambda snippet: snippet + ["<END>"]).values.tolist())
val_output_snippet_sequences = text2seq(df_val['clean_snippet'].apply(lambda snippet: snippet + ["<END>"]).values.tolist())

print(len(train_input_snippet_sequences),len(test_input_snippet_sequences),len(val_input_snippet_sequences))

print(len(train_intent_sequences),len(test_intent_sequences),len(val_intent_sequences))

print(len(train_output_snippet_sequences),len(test_output_snippet_sequences),len(val_output_snippet_sequences))

train_intent_sequences[0]

train_input_snippet_sequences[0]

train_output_snippet_sequences[0]

all_intent_seq = train_intent_sequences + test_intent_sequences + val_intent_sequences
all_input_snippet_seq = train_input_snippet_sequences + test_input_snippet_sequences + val_input_snippet_sequences
all_output_snippet_seq = train_output_snippet_sequences + test_output_snippet_sequences + val_output_snippet_sequences
print(len(all_intent_seq))
print(len(all_input_snippet_seq))
print(len(all_output_snippet_seq))

all_snippet_seq = all_input_snippet_seq + all_output_snippet_seq

maxlen_intent = max([len(x) for x in all_intent_seq])
maxlen_snippet = max([len(x) for x in all_snippet_seq])
print(maxlen_intent)
print(maxlen_snippet)

pad_length_intent = 35

"""#**GET THE ENCODER-DECODER INPUTS**"""

#encoder as prepadding the tokens
train_padded_intent_sequences = pad_sequences(train_intent_sequences,maxlen = pad_length_intent,padding= 'pre')
test_padded_intent_sequences = pad_sequences(test_intent_sequences,maxlen = pad_length_intent,padding= 'pre')
val_padded_intent_sequences = pad_sequences(val_intent_sequences,maxlen = pad_length_intent,padding= 'pre')

train_padded_intent_sequences[0]

#decoder input as post padding the tokens
train_padded_input_snippet_sequences = pad_sequences(train_input_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')
test_padded_input_snippet_sequences = pad_sequences(test_input_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')
val_padded_input_snippet_sequences = pad_sequences(val_input_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')

train_padded_input_snippet_sequences[0]

#decoder output as post padding the tokens
train_padded_output_snippet_sequences = pad_sequences(train_output_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')
test_padded_output_snippet_sequences = pad_sequences(test_output_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')
val_padded_output_snippet_sequences = pad_sequences(val_output_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')

train_padded_output_snippet_sequences[0]

y_train = df_train['class'].values.tolist()
y_test = df_test['class'].values.tolist()
y_val = df_val['class'].values.tolist()

"""# **Load the embeddings pretrained**"""

print("ENCODER INPUT INTENTS")
print(len(train_padded_intent_sequences))
print(len(test_padded_intent_sequences))
print(len(val_padded_intent_sequences))
print("The single sequence intent's length after padding")
print(len(train_padded_intent_sequences[0]))
print(len(test_padded_intent_sequences[0]))
print(len(val_padded_intent_sequences[0]))

print("DECODER INPUT INTENTS")
print(len(train_padded_input_snippet_sequences))
print(len(test_padded_input_snippet_sequences))
print(len(val_padded_input_snippet_sequences))
print("The single sequence input snippet's length after padding")
print(len(train_padded_input_snippet_sequences[0]))
print(len(test_padded_input_snippet_sequences[0]))
print(len(val_padded_input_snippet_sequences[0]))

print("DECODER OUTPUT INTENTS")
print(len(train_padded_output_snippet_sequences))
print(len(test_padded_output_snippet_sequences))
print(len(val_padded_output_snippet_sequences))
print("The single sequence output snippet's length after padding")
print(len(train_padded_output_snippet_sequences[0]))
print(len(test_padded_output_snippet_sequences[0]))
print(len(val_padded_output_snippet_sequences[0]))

print(len(y_train))
print(len(y_test))
print(len(y_val))

NUM_TRAIN = 56438
NUM_TEST = 15813
NUM_VAL = 6812

IN_LSTM_NODES = 100
OUT_LSTM_NODES = 100
EMBEDDING_SIZE = 100

"""## **Intents**"""

#load W2V vectors from the pretrained intent embeddings
model_filename = 'intents-w2v.model'
w2v_embedding = KeyedVectors.load(model_filename)
w2v_embedding

w2v_embedding.get_vector("how")

word2idx_inputs = tok_intent.word_index
len(word2idx_inputs)

num_words_intent = len(word2idx_inputs) + 1
print(num_words_intent)

embedding_matrix = np.zeros((num_words_intent, EMBEDDING_SIZE))
how_many_non_vocab = 0
for word, index in word2idx_inputs.items():
  try:
    embedding_vector = w2v_embedding.get_vector(word)
  except KeyError:
    embedding_vector = np.zeros((EMBEDDING_SIZE))
    how_many_non_vocab += 1
  
  embedding_matrix[index] = embedding_vector

how_many_non_vocab

embedding_matrix

"""## **Code Snippet (TO UPDATE for each MODEL (1-4))**"""

num_words_output = CODE_VOCAB_SIZE
num_words_output

p = str(Path.cwd())
glove_f_co = datapath(p + '/combined-glove-vectors-mincount0.txt')
tmp_f_co = get_tmpfile(p + '/combined-glove-vectors-as-w2v.txt')
_ = glove2word2vec(glove_f_co, tmp_f_co)
co_model_g = KeyedVectors.load_word2vec_format(tmp_f_co)

word2idx_outputs = updated_word_index

len(co_model_g.wv.vocab)

#del code_embedding_matrix

code_embedding_matrix = np.zeros((num_words_output + 1, EMBEDDING_SIZE))
how_many_non_vocab = 0
words = []
for word, index in word2idx_outputs.items():
  try:
    embedding_vector = co_model_g.get_vector(word)
  except KeyError:
    embedding_vector = np.zeros((EMBEDDING_SIZE)) #even OOV is mapped as a zeros vector
    how_many_non_vocab += 1
    words.append(word)
    
  
  code_embedding_matrix[index] = embedding_vector

how_many_non_vocab

words[:25]

"""# **MODELLING (BASE)**

##  **Seq2Seq Modelling**

Reference: https://github.com/samurainote/seq2seq_translate_slackbot/blob/master/seq2seq_translate.py
"""

"""
Encoder Architecture
"""

encoder_inputs = Input(shape=(pad_length_intent,))

#Our addition starts
embedding_layer = Embedding(num_words_intent, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=pad_length_intent)
embedding_layer.trainable = False
intent_sequence_w2v = embedding_layer(encoder_inputs)
#Our addition ends

encoder_lstm = LSTM(units=IN_LSTM_NODES, return_state=True, return_sequences=False)
# x-axis: time-step lstm
encoder_outputs, state_h, state_c = encoder_lstm(intent_sequence_w2v)
encoder_states = [state_h, state_c] # We discard `encoder_outputs` and only keep the states.

"""
Decoder Architecture
"""
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.

decoder_inputs = Input(shape = (1 + pad_length_snippet,))
code_embedding_layer = Embedding(num_words_output + 1,EMBEDDING_SIZE,weights = [code_embedding_matrix],input_length=pad_length_snippet)
code_embedding_layer.trainable = False
decoder_inputs_embed = code_embedding_layer(decoder_inputs)

decoder_lstm = LSTM(units=OUT_LSTM_NODES, return_sequences=True, return_state=True)
# x-axis: time-step lstm
decoder_outputs, de_state_h, de_state_c = decoder_lstm(decoder_inputs_embed,initial_state = encoder_states) #decoder_lstm(decoder_inputs, initial_state=encoder_states) # Set up the decoder, using `encoder_states` as initial state.
decoder_softmax_layer = Dense(num_words_output+1, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

"""
Encoder-Decoder Architecture
"""
# Define the model that will turn, `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.layers

model.compile(optimizer="Adam", loss="categorical_crossentropy",metrics = ['accuracy']) # Set up model

model.summary()

keras.utils.plot_model(model,show_shapes=True,show_layer_names=True,to_file="seq2seqmodel.png")

#INFERENCE SET UP
# inputs=encoder_inputs, outputs=encoder_states
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
encoder_model.summary()

# State from encoder
decoder_state_input_h = Input(shape=(OUT_LSTM_NODES,))
decoder_state_input_c = Input(shape=(OUT_LSTM_NODES,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = code_embedding_layer(decoder_inputs_single)

# x-axis: time-step lstm
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)

#decoder_outputs = argmax_layer(decoder_outputs)

decoder_model = Model(inputs=[decoder_inputs_single] + decoder_state_inputs, outputs=[decoder_outputs] + decoder_states)
decoder_model.summary()

print(decoder_state_input_c.shape)
print(decoder_state_input_h.shape)

print(decoder_outputs.shape)

"""## **Binary Classifier**"""

drop_rate = 0.5

intent_input = Input(EMBEDDING_SIZE,)
code_sequence = Input(EMBEDDING_SIZE,)

concat_input = keras.layers.Concatenate(axis=1)([intent_input,code_sequence])
hidden1 = Dense(100,activation="relu")(concat_input)
drop1 = keras.layers.Dropout(drop_rate)(hidden1)
hidden2 = Dense(50,activation="relu")(drop1)
drop2 = keras.layers.Dropout(drop_rate)(hidden2)
hidden3 = Dense(25,activation="relu")(drop2)
drop3 = keras.layers.Dropout(drop_rate)(hidden2)
output_layer = Dense(1,activation="sigmoid")(drop3)

binary_model = Model(inputs = [intent_input,code_sequence],outputs = [output_layer])
binary_model.compile(loss = 'binary_crossentropy',metrics = ['accuracy'],optimizer = "Adam")

binary_model.summary()

keras.utils.plot_model(binary_model,show_layer_names=True,show_shapes=True,to_file="binary_model.png")

"""#**FITTING models**

## **Fitting Seq2seq**
"""

where_train_positive = np.where(np.array(y_train) == 1)[0]
len(where_train_positive)

where_val_positive = np.where(np.array(y_val) == 1)[0]
len(where_val_positive)

where_test_positive = np.where(np.array(y_test) == 1)[0]
len(where_test_positive)

pos_train_intents = train_padded_intent_sequences[where_train_positive][:]
pos_test_intents = test_padded_intent_sequences[where_test_positive][:]
pos_val_intents = val_padded_intent_sequences[where_val_positive][:]

print(pos_train_intents.shape)
print(pos_test_intents.shape)
print(pos_val_intents.shape)

pos_train_snip_decinput = train_padded_input_snippet_sequences[where_train_positive][:]
pos_test_snip_decinput = test_padded_input_snippet_sequences[where_test_positive][:]
pos_val_snip_decinput = val_padded_input_snippet_sequences[where_val_positive][:]

print(pos_train_snip_decinput.shape)
print(pos_test_snip_decinput.shape)
print(pos_val_snip_decinput.shape)

pos_train_snip_decoutput = train_padded_output_snippet_sequences[where_train_positive][:]
pos_test_snip_decoutput = test_padded_output_snippet_sequences[where_test_positive][:]
pos_val_snip_decoutput = val_padded_output_snippet_sequences[where_val_positive][:]

print(pos_train_snip_decoutput.shape)
print(pos_test_snip_decoutput.shape)
print(pos_val_snip_decoutput.shape)

decoder_targets_train_one_hot = np.zeros((
         28520,
         pad_length_snippet + 1,
         num_words_output + 1
     ),
     dtype='float32'
 )

decoder_targets_test_one_hot = np.zeros((
         8127,
         pad_length_snippet + 1,
         num_words_output + 1
     ),
     dtype='float32'
 )

decoder_targets_val_one_hot = np.zeros((
         3439,
         pad_length_snippet + 1,
         num_words_output + 1
     ),
     dtype='float32'
 )
decoder_sample_val_one_hot = np.zeros((3,pad_length_snippet+1,num_words_output+1),dtype='float32')

for i, d in enumerate(pos_train_snip_decoutput):
    for t, word in enumerate(d):
      decoder_targets_train_one_hot[i, t, word] = 1

for i, d in enumerate(pos_test_snip_decoutput):
    for t, word in enumerate(d):
        decoder_targets_test_one_hot[i, t, word] = 1

for i, d in enumerate(pos_val_snip_decoutput):
    for t, word in enumerate(d):
      decoder_targets_val_one_hot[i, t, word] = 1

idx_word_intent = tok_intent.index_word
idx2word_target = {v:k for k, v in word2idx_outputs.items()}

for i,d in enumerate(pos_train_snip_decoutput[:3][:]):
  for t, idx in enumerate(d):
    decoder_sample_val_one_hot[i,t,idx] = 1

decoder_sample_val_one_hot.shape

sample_intent = pos_train_intents[:3][:]
sample_dec_input = pos_train_snip_decinput[:3][:]
print(sample_intent.shape)
print(sample_dec_input.shape)

sample_intent[0]

reshape_intent = sample_intent.reshape((3,35))
reshape_intent.shape

EPOCHS = 10

seq2seqmodel_history = model.fit(
          [pos_train_intents, pos_train_snip_decinput],
          decoder_targets_train_one_hot,
          validation_data= ([pos_val_intents,pos_val_snip_decinput],decoder_targets_val_one_hot),
          epochs=EPOCHS,
          batch_size = 256
          )

word2idx_outputs['<START>']

word2idx_outputs['<END>']

# getting the output from the seq2seq
# we are planning to get the word2vec embedding for the predicted code sequence from the seq2seq model
# we are adapting some of the stuff for a regular translation task

def code_generate(input_seq,_code_embedding_matrix = code_embedding_matrix):
  
    states_value = encoder_model.predict([input_seq])
    reshape_states_valueh = states_value[0][-1][:].reshape(1,-1) #just pass in the last hidden state
    reshape_states_valuec = states_value[1][-1][:].reshape(1,-1) #just pass in the last context state
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<START>']
    end_token = word2idx_outputs['<END>']
    output_sentence = []
    avg_embedding = np.zeros((EMBEDDING_SIZE,1),dtype='float32')

    for _ in range(pad_length_snippet):
        output_tokens, h, c = decoder_model.predict([target_seq,reshape_states_valueh,reshape_states_valuec])
        idx = np.argmax(output_tokens[0, 0, :])

        if end_token == idx:
             return ' '.join(output_sentence),avg_embedding

        word = ''

        if idx >= 0:
            word = idx2word_target[idx]
            output_sentence.append(word)
            vector = _code_embedding_matrix[idx].reshape(-1,1)
            avg_embedding += vector
            

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = idx
        states_value = [h, c]
      
    
    return ' '.join(output_sentence),avg_embedding/len(output_sentence)

with open('seq2seq_trainHistoryDict.pkl', 'wb') as file_pi:
  pickle.dump(seq2seqmodel_history.history, file_pi)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("seq2seq_model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("seq2seq_model.h5")
print("Saved model to disk")

"""## **Fitting on the binary model using the training samples**"""

def intent_w2v(sequence,_embedding_matrix = embedding_matrix):
  #embedding matrix: idx to vector mapping (it is a matrix)
  avg_embed = np.zeros((EMBEDDING_SIZE,1))
  num_nonzero = 0
  for token in sequence:
    if token != 0:
      vector = _embedding_matrix[token].reshape(-1,1)
      avg_embed += vector
      num_nonzero += 1
  
  return avg_embed/num_nonzero

def code_w2v(sequence, _embedding_matrix = code_embedding_matrix):
  #embedding matrix: idx to vector mapping (it is a matrix)
  avg_embed = np.zeros((EMBEDDING_SIZE,1))
  num_nonzero = 0
  for token in sequence:
    if token != 0:
      vector = _embedding_matrix[token].reshape(-1,1)
      avg_embed += vector
      num_nonzero += 1
  
  return avg_embed/num_nonzero

train_intent_embeds = np.zeros((NUM_TRAIN,EMBEDDING_SIZE))
train_code_embeds = np.zeros((NUM_TRAIN,EMBEDDING_SIZE))
print("Training code and intents")
for i,intent in enumerate(train_intent_sequences):
  intent_embed = intent_w2v(intent)
  code = train_padded_input_snippet_sequences[i][1:]
  code_embed = code_w2v(code)
  train_code_embeds[i][:] = code_embed[:][0]
  train_intent_embeds[i][:] = intent_embed[:][0]
  
  if i!= 0 and i%1000 == 0:
    print(f"{i}th iteration")


val_intent_embeds = np.zeros((NUM_VAL,EMBEDDING_SIZE))
val_code_embeds = np.zeros((NUM_VAL,EMBEDDING_SIZE))
i = 0
print("validation code and intents")
for intent in val_intent_sequences:
  intent_embed = intent_w2v(intent)
  code = val_padded_input_snippet_sequences[i][1:]
  code_embed = code_w2v(code)
  val_code_embeds[i][:] = code_embed[:][0]
  val_intent_embeds[i][:] = intent_embed[:][0]
  i += 1
  if i!= 0 and i%1000 == 0:
    print(f"{i}th iteration")

print(train_intent_embeds.shape,'\t',train_code_embeds.shape)
#print(test_intent_embeds.shape,'\t',test_code_embeds.shape)
print(val_intent_embeds.shape,'\t',val_code_embeds.shape)

#INPUTS TO THE BINARY MODEL
# INTENT W2V: avg W2V embedding for the intent sequence
# Code W2V/the average hidden state vector: the output of the Seq2seq model which needs to be applied 
# across all the samples in the training/val/test sets
'''
X: intent_embeds,code_embeds
y: the class label for the sample set working currently: y_train[:3]
'''
bin_history = binary_model.fit(x=[train_intent_embeds,train_code_embeds],
                               y=np.array(y_train),
                               validation_data=([val_intent_embeds,val_code_embeds],np.array(y_val)),
                               epochs = EPOCHS,
                                batch_size = 256
                               )

with open('binary_trainHistoryDict.pkl', 'wb') as file_pi:
  pickle.dump(bin_history.history, file_pi)
# serialize model to YAML
model_yaml = binary_model.to_yaml()
with open("binary_model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
binary_model.save_weights("binary_model.h5")
print("Saved model to disk")


# plotting
yaml_file = open('seq2seq_model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = keras.models.model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("seq2seq_model.h5")
print("Loaded model from disk")
#
#
#with open("seq2seq_trainHistoryDict.pkl",'rb') as f:
#  history = pickle.load(f)
#print(history)
#
#plt.figure()
#plt.plot(list(range(1,EPOCHS+1)),history['loss'],label='train')
#plt.plot(list(range(1,EPOCHS+1)),history['val_loss'],label = 'val')
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.legend()
#plt.title("Loss for Seq2Seq")
#plt.savefig("loss_seq2seq.png")
#
#plt.figure()
#plt.plot(range(1,EPOCHS+1),np.array(history['accuracy'])*100.00,label='train')
#plt.plot(range(1,EPOCHS+1),np.array(history['val_accuracy']) * 100.0,label = 'val')
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy (%)")
#plt.legend()
#plt.title("Accuracy for Seq2Seq")
#plt.savefig("acc_seq2seq.png")
#
loaded_model.compile(optimizer="Adam", loss="categorical_crossentropy",metrics = ['accuracy']) # Set up model
scores = loaded_model.evaluate([pos_test_intents,pos_test_snip_decinput],decoder_targets_test_one_hot)
print('Seq2Seq')
print("Test loss: ",scores[0])
print("Test accuracy: ",scores[1]*100.00)


test_intent_embeds = np.zeros((10,EMBEDDING_SIZE))# Change to 1000
test_code_embeds = np.zeros((10,EMBEDDING_SIZE))# Change to 1000
test_code_outputs = []
import time
t = int(time.time())
print("Testing code and intents")
y_test_sample = y_test[:10]# Change to 1000
print(dict(Counter(y_test_sample)))
i = 0
print("Testing code and intents")
for intent in test_intent_sequences[:10]:# Change to 1000
  intent_embed = intent_w2v(intent)
  code_output,code_embed = code_generate(intent)
  test_code_embeds[i][:] = code_embed[:][0]
  test_code_outputs.append(code_output)
  test_intent_embeds[i][:] = intent_embed[:][0]
  i += 1
  if i!= 0 and i%1000 == 0:
    print(f"{i}th iteration")
print(test_intent_embeds.shape,'\t',test_code_embeds.shape)

scores = binary_model.evaluate([test_intent_embeds,test_code_embeds],np.array(y_test_sample))
print("Binary Model Test loss: ",scores[0])
print("Binary Model Test accuracy: ",scores[1]*100.00)
y_pred = binary_model.predict([test_intent_embeds,test_code_embeds])
with open("y_true.pkl",'wb') as true:
  pickle.dump(y_test_sample,true)
with open("y_pred.pkl",'wb') as pred:
  pickle.dump(y_pred,pred)

