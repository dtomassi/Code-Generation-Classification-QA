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

    

    """
    Checking cosine similarity for this question.

    print("intent: " + str(question_ids[34705205][0]['intent']))
    snippet_one = question_ids[34705205][0]["snippet"]
    snippet_two = question_ids[34705205][1]["snippet"]
    corpus = [snippet_one, snippet_two]
    
    print(str(corpus))
    print("the vectors: " + str(get_vectors(corpus)))
    similarity_matrix = get_cosine_sim(corpus)
    """

    corpus = [x['snippet'] for x in question_ids[34705205]]

    #use TF using vectorization 
    #then, calculate cosine similarity after we get these vectors 

    
    
    # Sort answers by answer probability
    # Get Top n (e.g. 1,2,3) answers
    # Make sure the cosine sim. are not close (e.g. less than a threshold)
    # Return Top n answers that are not close as the answer to the question
    # print('similarity between snippet i: {} and snippet j: {} : {}'.format(i, j, similarity_matrix[i][j]))
    
  

 
    #questions_ids[34705205][i]['prob']
    # TODO: Refactor into function
    # TODO: Loop over all question IDs
    # question_dataset = []
    # for question_id in question_ids:
    #   question_dataset[question_id] = []
    # TODO: Write to new JSON file
    # TODO: Combine this new JSON file with train.json for a larger dataset
    
    corpus = sorted(question_ids[34705205], key=lambda x: x['prob'], reverse=True)
    similarity_matrix = get_cosine_sim([snippet['snippet'] for snippet in corpus])
    top_answers = [corpus[0]]

    print(str(similarity_matrix))

    N = 3
    SIM_THRESHOLD = 0.5
    PROB_THRESHOLD = 0.5
    for i in range(0, len(similarity_matrix[0])):
        if len(top_answers) == N:
            break
        for j in range(0, len(similarity_matrix)):
            if i != j:
                if len(top_answers) == N:
                    break
                #compare probabilities here
                #always append to top answers if less than similarity threshold
                
                if similarity_matrix[i][j] <= SIM_THRESHOLD:
                    #check probability of snippet j in the corpus 
                    top_answers.append(corpus[j])
                    
                # print("similarity between snippet i: " + str(i) + " and snippet j: " + str(j) + ": " + 
                # str(similarity_matrix[i][j]))
                # print("snippet i: " + corpus[i] + "snippet j: " + corpus[j])
                # print()
    print(top_answers)
    
    for answer in top_answers:
        if answer['prob'] < PROB_THRESHOLD
            top_answers.remove(answer)
    
    #could do another loop through top answers and remove the same answers and add unique answers to a new list.

    
    #1st, 3rd, 4th snippets (i-1)
    # corpus[0], corpus[2], corpus[3]

    # Iterate through answers
    # Compare probabilities to find top N
    

    # def find_top_n_prob(question_id, n):
    #   top_n = []
    #   top_n_lowest = 0
    #   for answer in question_id:
    #       if answer['prob] > top_n_lowest:
    #           REMOVE(top_n_lowest)
    #           ADD(answer)

 


def get_cosine_sim(corpus): 
    vectors = [t for t in get_vectors(corpus)]
    return cosine_similarity(vectors)
    
def get_vectors(corpus):
    #corpus = [t for t in snippets]
    vectorizer = CountVectorizer() 
    X = vectorizer.fit_transform(corpus)
    return X.toarray()



main()


