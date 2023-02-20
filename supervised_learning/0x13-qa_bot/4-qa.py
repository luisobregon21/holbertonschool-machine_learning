#!/usr/bin/env python3
''' QA chat bot '''
semantic_search = __import__('3-semantic_search').semantic_search
question_answer = __import__('0-qa').question_answer


def question_answer(corpus_path):
    '''
    answers questions from multiple reference texts
    :corpus_path: path to the corpus of reference documents
    '''
    words = ['bye', 'goodbye', 'quit', 'exit']
    loop = True
    while loop:
        user_input = input("Q: ")
        user_input = user_input.lower()
        if user_input.lower() in words:
            print("A: Goodbye")
            loop = False

        reference = semantic_search(corpus_path, user_input)
        answer = question_answer(user_input, reference)

        if answer == None or answer == "" or user_input in answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: ", answer)