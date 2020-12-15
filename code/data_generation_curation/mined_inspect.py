import json
from collections import Counter 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
# Change conala_mined_fp to where it is on your machine.


def main():
    conala_mined_fp = 'conala-mined.jsonl'
    with open(conala_mined_fp) as f:
        data = [json.loads(x) for x in f.readlines()]


    question_ids = {}
    for datum in data:
        q_id = datum['question_id']
        if q_id in question_ids:
            question_ids[q_id].append(datum)
        else:
            question_ids[q_id] = [datum]
    
    #print(str(question_ids[34705205]))
    print("intent: " + str(question_ids[34705205][0]['intent']))
    snippet_one = question_ids[34705205][0]["snippet"]
    snippet_two = question_ids[34705205][1]["snippet"]
    corpus = [snippet_one, snippet_two]
    print(str(corpus))
    #print("length is: " + str(len(question_ids.keys())))

    #use TF using vectorization 
    #then, calculate cosine similarity after we get these vectors 

    print("the vectors: " + str(get_vectors(corpus)))
    print("cosine similarity: " + str(get_cosine_sim(corpus)))


def get_cosine_sim(corpus): 
    vectors = [t for t in get_vectors(corpus)]
    return cosine_similarity(vectors)
    
def get_vectors(corpus):
    #corpus = [t for t in snippets]
    vectorizer = CountVectorizer() 
    X = vectorizer.fit_transform(corpus)
    return X.toarray()
    


main()


