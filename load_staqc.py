import pickle
import json
import re

#TODO: remove extra punctuation like ">>>, "", and things like "..." from snippet lists
#before checking the snippet size threshold

def main():

    #opening the questions file
    with open("python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle", 'rb') as task_file:
        task_dict = pickle.load(task_file)

    with open("python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle", 'rb') as snippet_file:
        snippet_dict = pickle.load(snippet_file)

    data_list = []

    count = 0
    cleaned_snippet = ""
    SNIPPET_SIZE_THRESHOLD = 5


    """opens the file and cleans the data, then, splits remaining data by newline,
    removes empty elements and whitespace. Then, checks if snippet less than or equal to threshold size"""

    with open("staqc.json", "w+") as data_file:
        for key, val in task_dict.items():

            snippet = snippet_dict[key]
            new_snippet = clean_data(snippet)

            snippet_list = snippet_dict[key].split("\n")
            new_snippet_list = new_snippet.split("\n")
            new_snippet_list = list(filter(None, new_snippet_list))
            for i in range(0, len(new_snippet_list)):
                new_snippet_list[i] = new_snippet_list[i].strip()


            if len(new_snippet_list) <= SNIPPET_SIZE_THRESHOLD:
                print("intent: " + str(val))
                print("snippet: " + str(snippet), end = '')
                if len(new_snippet_list) >= 1:
                    cleaned_snippet = "\n".join(new_snippet_list)
                    print("cleaned snippet: " + cleaned_snippet)

                    count += 1
                    print()
                    print()

                    #JSON object for each intent/snippet pair
                    data = {
                        "snippet": str(cleaned_snippet),
                        "intent": str(val),
                        "question_id": int(key)
                    }

                    data_list.append(data)


        json.dump(data_list, data_file, indent=4)


    print("snippets that can be used: " + str(count))
    data_file.close()





"""this function removes docstring comments, trailing and leading whitespaces and all one line comments."""
def clean_data(code_snippet):

    code_snippet = code_snippet.strip()
    code_snippet = re.sub("#.*", '', code_snippet)
    code_snippet = re.sub('"""[\s\S]*?"""', '', code_snippet)

    return code_snippet




main()
