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
if os.stat('save.p').st_size == 0:
    row = 0
else:
    row = pickle.load( open( "save.p", "rb" ) )



# Importing whole text column of data.csv into list
# TODO: make data.csv one column, or just use a text file
textList = list()
with open('data.csv', 'r') as csvFile:
    reader=csv.reader(csvFile)
    for line in reader:
        textList.append(line[1])

def label_sentences(text):
    html = markdown(text)
    plainText = ''.join(BeautifulSoup(html).findAll(text=True))
    paragraph = sent_tokenize(plainText)
    #print("paragraph 0: " + paragraph[0])
    index = 0
    while index < len(paragraph):
        print("\n" + paragraph[index] + "\n" + "sad[1], happy[2], anxiety[3], disgust[4], anger[5], surprise[6], neutral[7]")
        char = getch()
        if (char == "1"):
            emotion = "sadness"
        elif(char == "2"):
            emotion = "happiness"
        elif (char == "3"):
            emotion = "fear"
        elif (char == "4"):
            emotion = "disgust"
        elif (char == "5"):
            emotion = "anger"
        elif (char == "6"):
            emotion = "surprise"
        elif (char == "7"):
            emotion = "neutral"
        elif(char == "s"):
            emotion = "null"
        elif(char == "q"):
            exit(0)
        else:
            emotion = "null"
            index -= 1
            print("invalid input")
        if (emotion != "null"):
            row = [emotion,paragraph[index]]
            with open('data-labeled.csv','a') as fd:
                writer = csv.writer(fd)
                writer.writerow(row)
        index += 1

#MAIN CODE: loops through text list to label sentences
while row < len(textList):
    print("\n\nrow " + str(row) + ":\n\n")
    print(textList[row])
    char = getch()
    if (char == "y"):
        print("Yes!\n\n")
        label_sentences(textList[row])
        row += 1
        pickle.dump( row, open( "save.p", "wb" ) )
        time.sleep(button_delay)
    elif(char == "n"):
        print("No!\n\n")
        row += 1
        pickle.dump( row, open( "save.p", "wb" ) )
        time.sleep(button_delay)
    elif(char == "q"):
        pickle.dump( row, open( "save.p", "wb" ) )
        exit(0)
    else:
        print("Invalid Input!\n\n")
        time.sleep(button_delay)
