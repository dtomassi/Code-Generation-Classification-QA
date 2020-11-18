import pickle
import json

def main():

    #opening the questions file
    task_file = open("python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle", 'rb')
    snippet_file = open("python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle", 'rb')
    file_contents = []
    snippet_file_contents = []

    while True:
        try:
            file_contents.append(pickle.load(task_file))
            snippet_file_contents.append(pickle.load(snippet_file))
        except EOFError:
            break

    task_file.close()
    snippet_file.close()
    #print("file contents are: " + str(file_contents))

    #print(file_contents[0])
    task_dict = {}
    task_dict = file_contents[0]
    snippet_dict = {}
    snippet_dict = snippet_file_contents[0]
    data_list = []



    with open("staqc.json", "w+") as data_file:
        for key, val in task_dict.items():
            data = {
                "snippet": str(snippet_dict[key]),
                "intent": str(val),
                "question_id": int(key)
            }
            data_list.append(data)

        json.dump(data_list, data_file, indent=4)

    data_file.close()

    num_unique_questions = len(set(task_dict.keys()))
    print("the entire file has " + str(num_unique_questions) + " keys.")






main()
