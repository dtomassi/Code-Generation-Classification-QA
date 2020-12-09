# How to Use the Embeddings Models
## Word2Vec
To use either of the Word2Vec models using the gensim library, include the following:

```
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
```

Then load in the models with the following code:

***Combined Dataset Model:***

```
co_model_w2v = KeyedVectors.load('combined-w2v-model.bin')
```

***CodeSearchNet + Combined Dataset Model:***

```
csn_model_w2v_full = Word2Vec.load('csn-w2v.model')
csn_model_w2v = csn_model_w2v_full.wv # optional, but saves memory by converting the model to Key Vectors
```

### GloVe
To use either of the GloVe models using the gensim library, you must convert the GloVe vectors into Word2Vec. To start, include the following:

```
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from pathlib import Path
```

Then load in the models with the following code:

***Combined Dataset Model:***

```
p = str(Path.cwd())
glove_f_co = datapath(p + '/combined-glove-vectors.txt')
tmp_f_co = get_tmpfile(p + '/combined-glove-vectors-as-w2v.txt')

_ = glove2word2vec(glove_f_co, tmp_f_co)

co_model_g = KeyedVectors.load_word2vec_format(tmp_f_co)
```

***CodeSearchNet + Combined Dataset Model:***

```
glove_f_csn = datapath(p + '/csn-glove-vectors.txt')
tmp_f_csn = get_tmpfile(p + '/csn-glove-vectors-as-w2v.txt')

_ = glove2word2vec(glove_f_csn, tmp_f_csn)

csn_model_g = KeyedVectors.load_word2vec_format(tmp_f_csn)
```
