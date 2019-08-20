import csv
import pickle
import os
from itertools import islice
from helpers.getch import *
from helpers.getsentences import *
from nltk import sent_tokenize
from bs4 import BeautifulSoup
from markdown import markdown
button_delay = 0.2

#code to load row from save.p
def getRow():
    if os.stat('save.p').st_size == 0:
        return 0
    else:
        return pickle.load( open( "save.p", "rb" ) )

def getSentences(text):
    html = markdown(text)
    plainText = ''.join(BeautifulSoup(html).findAll(text=True))
    return sent_tokenize(plainText)

# Importing whole text column of data.csv into list
# TODO: make data.csv one column, or just use a text file
def makeTextList():
    textList = list()
    with open('../data/test.csv', 'r') as csvFile:
        reader=csv.reader(csvFile)
        for line in reader:
            textList.append(getSentences(line[0]))
    return textList

def label_sentences(text):
    index = 0
    while index < len(paragraph):
        print("\n" + paragraph[index] + "\n" + "positve[1], negative[2], neutral[3]")
        char = getch()
        if (char == "1"):
            sentiment = "positive"
        elif(char == "2"):
            sentiment = "negative"
        elif (char == "3"):
            sentiment = "neutral"
        elif(char == "s"):
            sentiment = "null"
        elif(char == "q"):
            exit(0)
        else:
            sentiment = "null"
            index -= 1
            print("invalid input")
        if (sentiment != "null"):
            row = [sentiment,paragraph[index]]
            with open('data-labeled.csv','a') as fd:
                writer = csv.writer(fd)
                writer.writerow(row)
        index += 1
if (__name__ == "__main__"):
    print(makeTextList())
