### Basic setup ##
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K
import numpy as np
import sys
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from pathlib import Path
import json
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict,Counter,OrderedDict

traces_filename = "overall-pos_neg_final-dataset.json"

CODE_VOCAB_SIZE = 5000
pad_length_intent = 35 #fixing the intent size as 35 from our experiments
pad_length_snippet = 50 #fixing the snippet size as 50 from our experiments

#define the number of samples for each set
NUM_TRAIN = 56438
NUM_TEST = 15813
NUM_VAL = 6812

#the hidden layer size for the LSTM and the embedding size we were using
IN_LSTM_NODES = 100
OUT_LSTM_NODES = 100
EMBEDDING_SIZE = 100
DROP_RATE = 0.5
EPOCHS = 25
BATCH_SIZE = 256
BIN_NUM_TEST = 1000

INTENT_MODEL_FILENAME = 'intents-w2v.model'
w2v_embedding = KeyedVectors.load(INTENT_MODEL_FILENAME)

if len(sys.argv) == 1:
  print("Wrong usage: please try: python3 embedding_avg_model.py <Embedding model name> <Type of embedding: W(word2vec)/G(Glove)>")
  exit(0)

argv = list(sys.argv[1:])
CODE_EMBEDDING_FILENAME = argv[0]
EMBEDDING_TYPE = argv[1]

if EMBEDDING_TYPE == 'W':
  w2v_full = Word2Vec.load(CODE_EMBEDDING_FILENAME)
  w2v_code = w2v_full.wv

elif EMBEDDING_TYPE == "G":
  p = str(Path.cwd())
  glove_f_co = datapath(p + "/" + CODE_EMBEDDING_FILENAME)
  tmp_f_co = get_tmpfile(p + '/glove-vectors-as-w2v.txt')
  _ = glove2word2vec(glove_f_co, tmp_f_co)
  w2v_code = KeyedVectors.load_word2vec_format(tmp_f_co)

else:
  print("WRONG USAGE of second argument: please try: python3 embedding_avg_model.py <Embedding model name> <Type of embedding: W(word2vec)/G(Glove)>")
  exit(0)

#loading the basic file
with open(traces_filename) as trace_file:
  json_traces = json.load(trace_file)

#we need to filter out the snippets with more than pad_length_snippet size
#we do this before we create the DataFrame
final_data = []
for datum in json_traces:
  snippet_length = len(datum['clean_snippet'])
  #ignore these long snippets
  if snippet_length > pad_length_snippet:
    continue
  
  else:
    final_data.append(datum) #we add the smaller length snippets

df = pd.DataFrame(final_data) #the dataframe


"""### TRAIN-TEST-VAL Split ###"""

"""80-20 Train-test split
and then last some of the train samples are validation samples
"""

df_train_all,df_test = train_test_split(df,test_size = 0.2,random_state = 999)


df_train = df_train_all.iloc[:-int(0.1*68128),:] 
df_val = df_train_all.iloc[-int(0.1*68128):,:]

"""### Custom tokenizer for the code snippets##"""

#we create the input and the output snippet sequences for the decoder model
input_decoder_snippets = df['clean_snippet'].apply(lambda snippet: ["<START>"] + snippet).values.tolist()
output_decoder_snippets = df['clean_snippet'].apply(lambda snippet: snippet + ["<END>"]).values.tolist()

#find the total vocab across the entire snippet sequences
vocab = []
for tokens in input_decoder_snippets + output_decoder_snippets:
  for token in tokens:
    vocab.append(token)


# we create a tokenizer on the lines of a keras tokenizer but we used our own because we have a list of snippet tokens
# rather than a string sequence. We create a dictionary of the code tokens by ordering them based of their vocab counts

word_counts = dict(Counter(vocab)) #create the count based dictionary first
wcounts = list(word_counts.items()) 
wcounts.sort(key=lambda x: x[1], reverse=True) #sort the dictionary based of its counts

sorted_voc = []
sorted_voc.extend(wc[0] for wc in wcounts) #get all the code tokens ordered based of the counts
word_index = OrderedDict(zip(sorted_voc, list(range(1,1 + len(sorted_voc))))) #create the tokenizer numbering integers ordering keys from its increasing count in decreasing order of their counts

updated_word_index = OrderedDict(list(word_index.items())[:CODE_VOCAB_SIZE]) #Pick the top CODE_VOCAB_SIZE many tokens for our corpus building

OOV = '<OOV>'
updated_word_index[OOV] = CODE_VOCAB_SIZE #map the word index as the last token for the OOV token

#the function to get the integer mapped sequence of all the tokens based of the word index we are using
def text2seq(snippets,w2i = updated_word_index, maxlen = CODE_VOCAB_SIZE):
  sequences = []
  #for each snippet
  for tokens in snippets:
    seq = []
    #for each token
    for token in tokens:
      #if token is found in the given w2i mapping
      if token in w2i:
        index = w2i[token] #map it accordingly
        seq.append(index)
      
      else:
        seq.append(w2i[OOV]) #else map it as OOV
    
    sequences.append(seq) #add the current sequence
  
  #list of lists of integer representation of the tokens
  return sequences

"""# **Tokenize the intent and the snippets**

Reference: https://stackabuse.com/python-for-nlp-neural-machine-translation-with-seq2seq-in-keras/
"""

tok_intent = Tokenizer() #use the keras tokenizer for the intents

input_encoder_intents = df['cleaned_intent'].values.tolist() #get the tokens first

tok_intent.fit_on_texts(input_encoder_intents) #fit on all the texts

#save the tokenizers first
with open("tok-intent.pkl",'wb') as f:
  pickle.dump(tok_intent,f)

with open("tok-snippet.pkl",'wb') as f:
  pickle.dump(word_index,f)

#apply the tokenizer on the train,val and the test sequences
train_intent_sequences = tok_intent.texts_to_sequences(df_train['cleaned_intent'].values.tolist())
test_intent_sequences = tok_intent.texts_to_sequences(df_test['cleaned_intent'].values.tolist())
val_intent_sequences = tok_intent.texts_to_sequences(df_val['cleaned_intent'].values.tolist())

#the text2seq function applies the tokenizer for the code which we learnt and it applies it accordingly to train,test and val for the decoder input code sequence
train_input_snippet_sequences = text2seq(df_train['clean_snippet'].apply(lambda snippet: ["<START>"] + snippet).values.tolist())
test_input_snippet_sequences = text2seq(df_test['clean_snippet'].apply(lambda snippet: ["<START>"] + snippet).values.tolist())
val_input_snippet_sequences = text2seq(df_val['clean_snippet'].apply(lambda snippet: ["<START>"] + snippet).values.tolist())

#the text2seq function applies the tokenizer for the code which we learnt and it applies it accordingly to train,test and val for the decoder output code sequence
train_output_snippet_sequences = text2seq(df_train['clean_snippet'].apply(lambda snippet: snippet + ["<END>"]).values.tolist())
test_output_snippet_sequences = text2seq(df_test['clean_snippet'].apply(lambda snippet: snippet + ["<END>"]).values.tolist())
val_output_snippet_sequences = text2seq(df_val['clean_snippet'].apply(lambda snippet: snippet + ["<END>"]).values.tolist())


"""#**GET THE ENCODER-DECODER INPUTS**"""

#encoder as prepadding the tokens is a general practice
train_padded_intent_sequences = pad_sequences(train_intent_sequences,maxlen = pad_length_intent,padding= 'pre')
test_padded_intent_sequences = pad_sequences(test_intent_sequences,maxlen = pad_length_intent,padding= 'pre')
val_padded_intent_sequences = pad_sequences(val_intent_sequences,maxlen = pad_length_intent,padding= 'pre')


#decoder input as post padding the tokens is another general practice
train_padded_input_snippet_sequences = pad_sequences(train_input_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')
test_padded_input_snippet_sequences = pad_sequences(test_input_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')
val_padded_input_snippet_sequences = pad_sequences(val_input_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')


#decoder output as post padding the tokens
train_padded_output_snippet_sequences = pad_sequences(train_output_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')
test_padded_output_snippet_sequences = pad_sequences(test_output_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')
val_padded_output_snippet_sequences = pad_sequences(val_output_snippet_sequences,maxlen = 1 + pad_length_snippet,padding= 'post')

#create the output labels right now for the binary classifier
y_train = df_train['class'].values.tolist()
y_test = df_test['class'].values.tolist()
y_val = df_val['class'].values.tolist()

"""# **Load the embeddings pretrained**"""

"""## **Intents**"""

#load W2V vectors from the pretrained intent embeddings
word2idx_inputs = tok_intent.word_index #the word index for the mapping

num_words_intent = len(word2idx_inputs) + 1 #the total vocab size of the intents in the W2V model (already pretrained for the intents alone)

embedding_matrix = np.zeros((num_words_intent, EMBEDDING_SIZE)) #load the embedding matrix internally
for word, index in word2idx_inputs.items():
  try:
    embedding_vector = w2v_embedding.get_vector(word) #get word if in the w2i mapping and it is captured
  except KeyError:
    embedding_vector = np.zeros((EMBEDDING_SIZE)) #fill with zeros for a non-existent word in the w2i mapping

  embedding_matrix[index] = embedding_vector #the vector is then added to the embedding matrix


"""## **Code Snippets**"""

num_words_output = CODE_VOCAB_SIZE
word2idx_outputs = updated_word_index #the word2index we had developed for the code tokens

code_embedding_matrix = np.zeros((num_words_output + 1, EMBEDDING_SIZE))

#all the code tokens in the wordindex
for word, index in word2idx_outputs.items():
  try:
    embedding_vector = w2v_code.get_vector(word)
  except KeyError:
    embedding_vector = np.zeros((EMBEDDING_SIZE)) #even OOV is mapped as a zeros vector
    
  
  code_embedding_matrix[index] = embedding_vector


"""# **MODELLING**

##  **Seq2Seq Modelling**

Reference: https://github.com/samurainote/seq2seq_translate_slackbot/blob/master/seq2seq_translate.py
"""

"""
Encoder Architecture
"""

#create the input for the encoder model: the intent padded sequence
encoder_inputs = Input(shape=(pad_length_intent,))

#Our addition starts
embedding_layer = Embedding(num_words_intent, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=pad_length_intent)
embedding_layer.trainable = False #freeze the embedding layer since it is already trained
intent_sequence_w2v = embedding_layer(encoder_inputs) #this is the output of the embedding layer
#Our addition ends

encoder_lstm = LSTM(units=IN_LSTM_NODES, return_state=True, return_sequences=False) #the LSTM comes in next
encoder_outputs, state_h, state_c = encoder_lstm(intent_sequence_w2v)
encoder_states = [state_h, state_c] # We discard `encoder_outputs` and only keep the states.

"""
Decoder Architecture
"""
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.

decoder_inputs = Input(shape = (1 + pad_length_snippet,)) #we will extra layer for the input since we also have <START> <END> tokens
code_embedding_layer = Embedding(num_words_output + 1,EMBEDDING_SIZE,weights = [code_embedding_matrix],input_length=pad_length_snippet)
code_embedding_layer.trainable = False #again freeze the layers
decoder_inputs_embed = code_embedding_layer(decoder_inputs)

decoder_lstm = LSTM(units=OUT_LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, de_state_h, de_state_c = decoder_lstm(decoder_inputs_embed,initial_state = encoder_states) #decoder_lstm(decoder_inputs, initial_state=encoder_states) # Set up the decoder, using `encoder_states` as initial state.
decoder_softmax_layer = Dense(num_words_output+1, activation='softmax') #predict the output words
decoder_outputs = decoder_softmax_layer(decoder_outputs) #it will be predicting the code token accordingly

"""
Encoder-Decoder Architecture
"""
# Define the model that will turn, `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

decoder_layer = model.layers[-2] #grab the decoder layer which is the second last layer

model.compile(optimizer="Adam", loss="categorical_crossentropy",metrics = ['accuracy']) # Set up model
print("Seq2Seq model\n")
model.summary()

#save the model architecture
keras.utils.plot_model(model,show_shapes=True,show_layer_names=True,to_file="seq2seqmodel.png")

#INFERENCE SET UP
# inputs=encoder_inputs, outputs=encoder_states for the inference phase
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
print("ENCODER MODEL\n")
encoder_model.summary()

# State from encoder
decoder_state_input_h = Input(shape=(OUT_LSTM_NODES,))
decoder_state_input_c = Input(shape=(OUT_LSTM_NODES,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,)) #predict word by word in inference phase
decoder_inputs_single_x = code_embedding_layer(decoder_inputs_single)

# x-axis: time-step lstm
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)

decoder_model = Model(inputs=[decoder_inputs_single] + decoder_state_inputs, outputs=[decoder_outputs] + decoder_states) #we need the states also for the next word prediction accordingly
print("DECODER MODEL\n")
decoder_model.summary()


"""## **Binary Classifier**"""

#we will pass in the average embedded w2v input for the input intent sequence; same for the code sequence predicted from the Seq2Seq model
intent_input = Input(EMBEDDING_SIZE,)
code_sequence = Input(EMBEDDING_SIZE,)

concat_input = keras.layers.Concatenate(axis=1)([intent_input,code_sequence]) #join the two inputs first
hidden1 = Dense(100,activation="relu")(concat_input)
drop1 = keras.layers.Dropout(DROP_RATE)(hidden1)
hidden2 = Dense(50,activation="relu")(drop1)
drop2 = keras.layers.Dropout(DROP_RATE)(hidden2)
hidden3 = Dense(25,activation="relu")(drop2)
drop3 = keras.layers.Dropout(DROP_RATE)(hidden2)
output_layer = Dense(1,activation="sigmoid")(drop3)

binary_model = Model(inputs = [intent_input,code_sequence],outputs = [output_layer])
binary_model.compile(loss = 'binary_crossentropy',metrics = ['accuracy'],optimizer = "Adam")
print("BINARY MODEL\n")
binary_model.summary()

keras.utils.plot_model(binary_model,show_layer_names=True,show_shapes=True,to_file="binary_model.png")

"""#**FITTING models**

## **Fitting Seq2seq**
"""

#find the positive samples within the sets first
#we only use positive samples for the seq2seq model alone
where_train_positive = np.where(np.array(y_train) == 1)[0]
where_val_positive = np.where(np.array(y_val) == 1)[0]
where_test_positive = np.where(np.array(y_test) == 1)[0]


POS_TRAIN = len(where_train_positive)
POS_VAL = len(where_val_positive)
POS_TEST = len(where_test_positive)


#get the intents for these samples alone
pos_train_intents = train_padded_intent_sequences[where_train_positive][:]
pos_test_intents = test_padded_intent_sequences[where_test_positive][:]
pos_val_intents = val_padded_intent_sequences[where_val_positive][:]


#do the same for the decoder input sequences
pos_train_snip_decinput = train_padded_input_snippet_sequences[where_train_positive][:]
pos_test_snip_decinput = test_padded_input_snippet_sequences[where_test_positive][:]
pos_val_snip_decinput = val_padded_input_snippet_sequences[where_val_positive][:]

#do the same for the decoder output sequences
pos_train_snip_decoutput = train_padded_output_snippet_sequences[where_train_positive][:]
pos_test_snip_decoutput = test_padded_output_snippet_sequences[where_test_positive][:]
pos_val_snip_decoutput = val_padded_output_snippet_sequences[where_val_positive][:]

#The One hot encoded sequence for the seq2seq prediction
#we pad an extra token for the END TOKEN sequence; num_words_output is also added so that we accomodate the prepad token
decoder_targets_train_one_hot = np.zeros((
         POS_TRAIN,
         pad_length_snippet + 1,
         num_words_output + 1
     ),
     dtype='float32'
 )

decoder_targets_test_one_hot = np.zeros((
         POS_TEST,
         pad_length_snippet + 1,
         num_words_output + 1
     ),
     dtype='float32'
 )

decoder_targets_val_one_hot = np.zeros((
         POS_VAL,
         pad_length_snippet + 1,
         num_words_output + 1
     ),
     dtype='float32'
 )


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
idx2word_target = {v:k for k, v in word2idx_outputs.items()} #for the code_generate phase, remap to the actual sequence


seq2seqmodel_history = model.fit(
          [pos_train_intents, pos_train_snip_decinput],
          decoder_targets_train_one_hot,
          validation_data= ([pos_val_intents,pos_val_snip_decinput],decoder_targets_val_one_hot),
          epochs=EPOCHS,
          batch_size = BATCH_SIZE
          )

with open('seq2seq_trainHistoryDict.pkl', 'wb') as file_pi:
  pickle.dump(seq2seqmodel_history.history, file_pi)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("seq2seq_model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("seq2seq_model.h5")
print("Seq2Seq Model Saved model to disk")


yaml_file = open('seq2seq_model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = keras.models.model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("seq2seq_model.h5")
print("Loaded Seq2Seq model from disk")

loaded_model.compile(optimizer="Adam", loss="categorical_crossentropy",metrics = ['accuracy']) # Set up model again
scores = loaded_model.evaluate([pos_test_intents,pos_test_snip_decinput],decoder_targets_test_one_hot)
print('Seq2Seq model')
print("Test loss: ",scores[0])
print("Test accuracy: ",scores[1]*100.00)


"""## **Fitting on the binary model using the training samples**"""

#the function of the hidden and context vectors are retrieved accordingly
hidden_layer_function = K.function([model.inputs], [decoder_layer.output])


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

#get the h,c for each dataset
train_out = hidden_layer_function([[train_padded_intent_sequences,train_padded_input_snippet_sequences],decoder_targets_train_one_hot])

h,c = train_out[0][0][1],train_out[0][0][2] #hidden state and context vectors

print("train embedding creating")
train_code_embeds = np.multiply(h,c) #element wise product of the h,c which is the input to the model
print("train Code embeds done")
train_intent_embeds = np.zeros((NUM_TRAIN,EMBEDDING_SIZE))
for i,intent in enumerate(train_intent_sequences):
  intent_embed = intent_w2v(intent)
  train_intent_embeds[i][:] = intent_embed[:][0]
  
  if i!= 0 and i%1000 == 0:
    print(i,"th iteration")

val_out = hidden_layer_function([[val_padded_intent_sequences,val_padded_input_snippet_sequences],decoder_targets_val_one_hot])

h,c = val_out[0][0][1],val_out[0][0][2]
val_code_embeds = np.multiply(h,c) #element wise product of the h,c which is the input to the model
print("validation code embedding done")
val_intent_embeds = np.zeros((NUM_VAL,EMBEDDING_SIZE))
i = 0
print("validation code and intents")
for intent in val_intent_sequences:
  intent_embed = intent_w2v(intent)
  val_intent_embeds[i][:] = intent_embed[:][0]
  i += 1
  if i!= 0 and i%1000 == 0:
    print(i,"th iteration")


test_out = hidden_layer_function([[test_padded_intent_sequences[:BIN_NUM_TEST][:],test_padded_input_snippet_sequences[:BIN_NUM_TEST][:]],decoder_targets_test_one_hot[:BIN_NUM_TEST][:][:]])
h,c = test_out[0][0][1],test_out[0][0][2]

test_intent_embeds = np.zeros((BIN_NUM_TEST,EMBEDDING_SIZE))
test_code_embeds = np.multiply(h,c) #element wise product of the h,c which is the input to the model
i = 0
print("Testing code and intents")
for intent in test_intent_sequences[:BIN_NUM_TEST]:
  intent_embed = intent_w2v(intent)
  test_intent_embeds[i][:] = intent_embed[:][0]
  i += 1
  if i!= 0 and i%1000 == 0:
    print(i,"th iteration")


#INPUTS TO THE BINARY MODEL
# INTENT W2V: avg W2V embedding for the intent sequence
# Code W2V/the average hidden state vector: the output of the Seq2seq model which needs to be applied 
# across all the samples in the training/val/test sets
'''
X: intent_embeds,code_embeds
y: the class labels

'''
bin_history = binary_model.fit(x=[train_intent_embeds,train_code_embeds],
                               y=np.array(y_train),
                               validation_data=([val_intent_embeds,val_code_embeds],np.array(y_val)),
                               epochs = EPOCHS,
                                batch_size = BATCH_SIZE
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


scores = binary_model.evaluate([test_intent_embeds,test_code_embeds],np.array(y_test[:BIN_NUM_TEST]))
print("Binary Test loss: ",scores[0])
print("Binary Test accuracy: ",scores[1]*100.00,"%")

y_pred = binary_model.predict([test_intent_embeds,test_code_embeds])

with open("y_true.pkl",'wb') as true:
  pickle.dump(y_test[:BIN_NUM_TEST],true)

with open("y_pred.pkl",'wb') as pred:
  pickle.dump(y_pred,pred)