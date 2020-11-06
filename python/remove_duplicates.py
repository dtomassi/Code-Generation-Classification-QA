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

    curated_answers = []
    no_answer_questions = 0
    count = 0
    for key, val in question_ids.items():
        top_answers = remove_duplicate_answers(question_ids[key])
        print("top answers: " + str(top_answers))
        #print("similar answers that were removed: " + str(similar_answers))
        print()
        if len(top_answers) == 0:
            no_answer_questions += 1
        else:
            curated_answers.append(top_answers)

    print('Number of answers processed: ', len(curated_answers))
    print('Number of no answer questions: ', no_answer_questions)
    with open('curated_mined.json', 'w+') as curated_file:
        json.dump(curated_answers, curated_file, indent=4)


def remove_duplicate_answers(answers):
    corpus = [x['snippet'] for x in answers]
    corpus = sorted(answers, key=lambda x: x['prob'], reverse=True)

    similarity_matrix = get_cosine_sim([snippet['snippet'] for snippet in corpus])
    top_answers = [corpus[0]]
    similar_answers = []

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
                # Compare similarities here.
                # Always append to top answers if less than similarity threshold
                if similarity_matrix[i][j] <= SIM_THRESHOLD:
                    top_answers.append(corpus[j])
                else:
                    similar_answers.append(corpus[j])

    for answer in top_answers:
        if answer['prob'] < PROB_THRESHOLD:
            top_answers.remove(answer)

    return top_answers


def get_cosine_sim(corpus):
    vectors = [t for t in get_vectors(corpus)]
    return cosine_similarity(vectors)

def get_vectors(corpus):
    #corpus = [t for t in corpus]
    print("corpus: " + str(corpus))
    vectorizer = CountVectorizer()

    try:
        X = vectorizer.fit_transform(corpus)

    except ValueError as e:
        print("we have a value error- no vocab detected")
        corpus_vocab = list(set(corpus))
        print("the new corpus without duplicate vocab")
        new_vectorizer = CountVectorizer(vocabulary=corpus_vocab)
        Y = new_vectorizer.fit_transform(corpus)
        print("features: " + str(new_vectorizer.get_feature_names()))
        return Y.toarray()


    print("features: " + str(vectorizer.get_feature_names()))
    return X.toarray()



main()



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
