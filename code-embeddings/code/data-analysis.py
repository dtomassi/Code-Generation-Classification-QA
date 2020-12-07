from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from pathlib import Path

### Load in Models ###

# Combined Dataset - w2v Model #
co_model_w2v = KeyedVectors.load('co-w2v-sg.bin')

# CodeSearchNet + Combined Dataset - w2v Model #
csn_model_w2v_full = Word2Vec.load('w2v-csn.model')
csn_model_w2v = csn_model_w2v_full.wv

# Combined Dataset - glove Model #
p = str(Path.cwd())
glove_f_co = datapath(p + '/vectors-co.txt')
tmp_f_co = get_tmpfile(p + '/glove_w2v_co.txt')

_ = glove2word2vec(glove_f_co, tmp_f_co)

co_model_g = KeyedVectors.load_word2vec_format(tmp_f_co)

# CodeSearchNet + Combined Dataset - glove Model #
glove_f_csn = datapath(p + '/vectors-csn.txt')
tmp_f_csn = get_tmpfile(p + '/glove_w2v_csn.txt')

_ = glove2word2vec(glove_f_csn, tmp_f_csn)

csn_model_g = KeyedVectors.load_word2vec_format(tmp_f_csn)


### Function that does the following:
###     1. Writes vocab count to the text file "vocab-count.txt"
###     2. Creates a bar graph of the 10 most frequent tokens
###     3. Creates 3 Possible PCA graphs: random tokens, all tokens, and most frequent tokens

def Analysis(model,color,label,l1,l2):
    # Sorts models by freqency of tokens #
    w2vSorted=dict(sorted(model.vocab.items(), key=lambda x: x[1],reverse=True))

    # Grabs 10 most frequent tokens #
    topten = 0
    freq_words = []
    freq_count = []
    v_info = []

    v_info.append('Information for ' + label + ':')
    print('number of words for ' + label + ': ' + str(len(model.vocab)))
    v_info.append('Number of words: ' + str(len(model.vocab)))

    if (l1=='g'):
        file = 'vocab-' + l2 + '.txt'
        tfile = open(file, 'r')
        g_count = {}
        temp = []
        for l in tfile:
            temp = l.split()
            g_count[temp[0]] = temp[1]

    def counter(w,m=model,l1=l1,l2=l2):
        if(l1=='w'):
            return m.vocab[w].count
        else:
            return int(g_count[w])

    v_info.append('\n10 Most Frequent Words + Count:')
    for words in w2vSorted.keys():
        if topten < 10:
            v_info.append(words + ': ' + str(counter(words)))
            print(words + ': ' + str(counter(words)))
            freq_words.append(words)
            freq_count.append(counter(words))
            topten += 1

    # Writes Vocab Information to vocab-results.txt #
    f = open("vocab-results.txt", "a")
    for r in v_info:
        f.write(r + "\n")
    f.write('\n\n')
    f.close()

    # Naming Scheme #
    if l1=='w':
        im_name = l2 + '-w2v'
    else:
        im_name = l2 + '-glove'

    # Creates bargraph of 10 most frequent tokens #
    x = freq_words
    count = freq_count

    fig = plt.figure(figsize=[9,4.5])
    plt.bar(x,count,color=color)
    plt.xlabel('Tokens')
    plt.ylabel('Token Count')
    plt.title('Ten Most Frequent Tokens: ' + label)
    name = 'FreqCount-' + im_name
    plt.savefig(name)

    #plt.show()

    w_size = len(model.vocab)

    ### Creates PCA Graphs ###
    ### parameters:
    ###     1. r: plots r random tokens
    ###     2. all: when set to True, plots all tokens
    ###     3. top: plots top most frequent tokens
    def setUp(r=0,all=False,top=0):
        words = list(model.vocab)

        ## Use to show first 25 tokens ##
        if r > 0 and r < w_size:
            random.seed(0)
            ## Create random token list ###
            word_list = []
            index = random.sample(range(w_size-1),r)
            for i in index:
                word_list.append(words[i])

            x_rand = []
            for w in word_list:
                x_rand.append(model[w])

            pca = PCA(n_components=2)
            result = pca.fit_transform(x_rand)
            plt.figure(figsize=(7,4))
            plt.scatter(result[0:, 0], result[0:, 1], c=color)

            for i, word in enumerate(word_list):
                plt.annotate(word, xy=(result[i,0], result[i,1]), fontsize='small')

            plt.title('PCA Graph for Random Tokens: ' + label)
            name = 'PCArandom-' + im_name
            plt.savefig(name)

            #plt.show()

        if all:
            ## Use for all vocab
            x = model[model.vocab]
            pca = PCA(n_components=2)
            result = pca.fit_transform(x)

            plt.figure(figsize=(7,4))
            plt.scatter(result[0:, 0], result[0:, 1], c=color, marker='.')
            words = list(model.vocab)
            #random.seed(0)
            """
            wl = []
            index = random.sample(range(w_size-1),15)
            for i in index:
                wl.append(words[i])

            for i, word in enumerate(wl):
                plt.annotate(word, xy=(result[index[i],0], result[index[i],1]), fontsize='small')
            """
            plt.title('PCA Graph for All Tokens: ' + label)
            name = 'PCAall-' + im_name
            plt.savefig(name)

            #plt.show()

        ## Use to show "top" most Frequent words ##
        if top > 0 and top < w_size:
            t = 0
            for words in w2vSorted.keys():
                if t < top:
                    freq_words.append(words)
                    t += 1
                else:
                    break

            x_freq = []
            for w in freq_words:
                x_freq.append(model[w])
            pca = PCA(n_components=2)
            result = pca.fit_transform(x_freq)
            plt.figure(figsize=(7,4))
            plt.scatter(result[0:,0], result[0:,1], c=color)

            for i, word in enumerate(freq_words):
                plt.annotate(word, xy=(result[i,0], result[i,1]), fontsize='small')

            plt.title('PCA Graph for Most Frequent Tokens: ' + label)
            name = 'PCAfreq-' + im_name
            plt.savefig(name)

            #plt.show()

    setUp(all=True,top=10)

# Combined Dataset - w2v Model #
Analysis(co_model_w2v,'firebrick','Combined Corpus (w2v)','w','co')

# CodeSearchNet + Combined Dataset - w2v Model #
Analysis(csn_model_w2v,'purple','CodeSearchNet + Combined Corpus (w2v)','w','csn')

# Combined Dataset - glove Model #
Analysis(co_model_g,'tomato','Combined Corpus (GloVe)','g','co')

# CodeSearchNet + Combined Dataset - glove Model #
Analysis(csn_model_g,'cadetblue','CodeSearchNet + Combined Corpus (GloVe)','g','csn')
