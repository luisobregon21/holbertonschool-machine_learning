#!/usr/bin/env python3
''' QA BOT '''
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
  '''
  answers questions from a reference text
  :reference: is the reference text
  '''
  words = ['bye', 'goodbye', 'quit', 'exit']
  loop = True
  while loop:
    user_input = input("Q: ")
    if user_input.lower() in words:
      print('A: Goodbye')
      loop = False

    answer = question_answer(user_input, reference)

    if answer == None or answer == "" or user_input in answer:
      print("A: Sorry, I do not understand your question.")
    else:
      print('A: ', answer)
