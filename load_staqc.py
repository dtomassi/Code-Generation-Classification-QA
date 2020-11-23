import pickle
import json
import re


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

    with open("new_staqc.json", "w+") as data_file:
        for key, val in task_dict.items():

            snippet = snippet_dict[key]
            new_snippet = clean_data(snippet)

            snippet_list = snippet_dict[key].split("\n")
            new_snippet_list = new_snippet.split("\n")
            new_snippet_list = list(filter(None, new_snippet_list))
            for i in range(0, len(new_snippet_list)):
                new_snippet_list[i] = new_snippet_list[i].strip()


            if len(new_snippet_list) <= SNIPPET_SIZE_THRESHOLD:
                if len(new_snippet_list) >= 1:
                    cleaned_snippet = "\n".join(new_snippet_list)
                    count += 1

                    #JSON object for each intent/snippet pair
                    data = {
                        "snippet": str(cleaned_snippet),
                        "intent": str(val),
                        "question_id": int(key)
                    }

                    data_list.append(data)


        json.dump(data_list, data_file, indent=4)


    print("snippets that can be used: " + str(count))


"""this function removes docstring comments, trailing and leading whitespaces and all one line comments."""
def clean_data(code_snippet):
    # Remove comments
    code_snippet = code_snippet.strip()
    code_snippet = re.sub("#.*", '', code_snippet)
    code_snippet = re.sub('"""[\s\S]*?"""', '', code_snippet)

    # Remove >>> and ... and returns
    # Example:
    # >>> for i in range(3):
    # ...   print(i)
    # 0
    # 1
    # 2
    # Becomes:
    # for i in range(3):
    #   print(i)

    # Arrow and dot removal lambdas
    arrow_removal = lambda x: re.sub('^\s*>>>\s*', '', x)
    dot_removal = lambda x: re.sub('\s*^\.\.\.\s*', '', x)

    # Look to see if using interpreter format and remove value return lines.
    # Example:
    # >>> 1 + 1
    # 2

    code_snippet_split = code_snippet.split('\n')
    cleaned_snippet_list = [code_snippet_split[0]]
    if re.search('^\s*>>>\s*', code_snippet_split[0]):
        for line in code_snippet_split[1:]:
            if re.search('^\s*>>>\s*|\s*^\.\.\.\s*', line):
                cleaned_snippet_list.append(line)
    cleaned_snippet = '\n'.join(cleaned_snippet_list)

    # Remove >>>
    cleaned_snippet = '\n'.join(list(filter(lambda x: x != '' and x != ' ', [arrow_removal(x) for x in cleaned_snippet.split('\n')])))
    if cleaned_snippet != code_snippet:
        cleaned_snippet = '\n'.join(list(filter(lambda x: x != '' and x != ' ', [dot_removal(x) for x in cleaned_snippet.split('\n')])))

    # If >>> is removed then remove ...
    if cleaned_snippet != code_snippet:
        cleaned_snippet_split = cleaned_snippet.split('\n')

    return cleaned_snippet


main()
