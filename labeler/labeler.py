import csv
import pickle
import os
from itertools import islice
from helpers.getch import *
from helpers.getsentences import *
from nltk import sent_tokenize
from bs4 import BeautifulSoup
from markdown import markdown
import random
button_delay = 0.2


def makeTextList():
    textList = list()
    with open('../data/data.csv', 'r') as csvFile:
        reader=csv.reader(csvFile)
        for line in reader:
            sentences = getSentences(line[0])
            for sentence in sentences:
                textList.append(sentence)
    return textList

#code to load row from save.p
def getList():
    if os.stat('save.p').st_size == 0:
        return makeTextList()
    else:
        return pickle.load( open( "save.p", "rb" ) )

def getSentences(text):
    html = markdown(text)
    plainText = ''.join(BeautifulSoup(html).findAll(text=True))
    return sent_tokenize(plainText)

# Importing whole text column of data.csv into list
# TODO: make data.csv one column, or just use a text file


def label_sentences(sentences):
    while (len(sentences) != 0):
        index = random.randint(0, len(list) - 1)
        print(len(sentences))
        print("\n" + sentences[index] + "\n" + "positve[1], negative[2], neutral[3]")
        char = getch()
        if (char == "1"):
            sentiment = "positive"
        elif(char == "2"):
            sentiment = "negative"
        elif (char == "3"):
            sentiment = "neutral"
        elif(char == "s"):
            del(sentences[index])
            continue
        elif(char == "q"):
            pickle.dump(sentences, open( "save.p", "wb" ))
            exit(0)
        else:
            print("invalid input")
            continue

        row = [sentiment,sentences[index]]
        with open('../data/data-labeled.csv','a') as fd:
            writer = csv.writer(fd)
            writer.writerow(row)
        del(sentences[index])

if (__name__ == "__main__"):
     list = getList()
     label_sentences(list)
