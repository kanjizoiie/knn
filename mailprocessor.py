##
# This file will eventually calculate the input values from a JSON file of mails.
#


import json
import io
import jsonloader

mails = jsonloader.load_file("mail.json")
words = jsonloader.load_file("words.json")

def count_words():
    for mail in mails["mails"]:
        for c in mail["content"]:
            print(c)
count_words()