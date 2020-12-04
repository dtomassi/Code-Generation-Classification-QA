import pickle
from gensim.models import Word2Vec

def main():

    intent_dict = []
    with open("intents_parsed.pkl",'rb') as intent_file:
	       intents_tokenized = pickle.load(intent_file)

    #split each string into a list of its own by using space as delim
    #create a list of these lists
    #put into word2vec

    word_tokenization = []

    for i in range(0, len(intents_tokenized)):
        snippet_word_tokenized = intents_tokenized[i].split()
        word_tokenization.append(snippet_word_tokenized)

    print(str(word_tokenization))
    model = Word2Vec(word_tokenization, min_count=1)

    #summarize vocab
    words = list(model.wv.vocab)

    #print(model['numpi', 'syntaxidiom', 'cast', 'n', 'n', '1'])

    #save model
    model.save('model.bin')
    new_model = Word2Vec.load('model.bin')
    print(new_model)







main()
