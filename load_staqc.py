import pickle
import json
import re

#add to a function

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

    count = 0
    cleaned_snippet = ""


    """remove newline chars, check if snippet only length 5 or less
    remove comments, empty strings"""

    with open("staqc.json", "w+") as data_file:
        for key, val in task_dict.items():

            #snippet_separate = list(filter(None, snippet_separate))
            snippet = snippet_dict[key]
            new_snippet = clean_data(snippet)

            #split the list, remove empty elements, and leading and trailing whitespace, and
            #pure whitespace
            snippet_list = snippet_dict[key].split("\n")
            new_snippet_list = new_snippet.split("\n")
            new_snippet_list = list(filter(None, new_snippet_list))
            for i in range(0, len(new_snippet_list)):
                new_snippet_list[i] = new_snippet_list[i].strip()



            if len(new_snippet_list) <= 5:
                print("intent: " + str(val))
                print("snippet: " + str(snippet), end = '')
                if len(new_snippet_list) >= 1:
                    cleaned_snippet = "\n".join(new_snippet_list)
                    print("cleaned snippet: " + cleaned_snippet)

                    count += 1
                    print()
                    print()


                    data = {
                        "snippet": str(cleaned_snippet),
                        "intent": str(val),
                        "question_id": int(key)
                    }

                    data_list.append(data)


        json.dump(data_list, data_file, indent=4)


    print("snippets that can be used: " + str(count))
    data_file.close()

#print("before filtering: " + snippet_dict[key], end = '')

#snippet_together = "\n".join(new_snippet_separate)
#print("snippet after filtering: " + snippet_together)
#print()


def clean_data(code_snippet):

    #removes docstring comments- fix up
    #removes all trailing and leading whitespaces
    #removes all one line comments

    code_snippet = code_snippet.strip()
    code_snippet = re.sub("#.*", '', code_snippet)
    code_snippet = re.sub('"""[\s\S]*?"""', '', code_snippet)

    return code_snippet




main()
