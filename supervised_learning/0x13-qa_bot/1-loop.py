#!/usr/bin/env python3
''' a script that takes in input from the user with the prompt Q: and prints A: as a response '''

words = ['bye', 'goodbye', 'quit', 'exit']
loop = True
while loop:
    user_input = input("Q: ")
    if user_input.lower() in words:
        print('A: Goodbye')
        loop = False
    else:
        print('A: ')