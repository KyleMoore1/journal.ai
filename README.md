# journal.ai
journal.ai is a NLP sentiment analysis application for journal entries. It currently classifies sentences from journal entries as `positive`, `negative`, or `neutral` with 67% accuracy on the validation set. This is a work in progress but the end goal is an app where people can write daily reflections and be given a percentage score of how positive/negative their reflection was. I hope by using this app, people will be motivated to reflect on their days in a more positive light, even on bad days. Included in this repo so far is a scraper that scrapes all sentences from a reddit community into a csv, a CLI labeler tool to label those sentences, and a jupyter notebook where I trained the classifier.

## Status of the project
Currently, I am in the labeling phase of the project. I have a csv with 3,634 sentences from diary entries labeled as positive, negative, or neutral. I trained an initial model with this data but it clearly overfits and I was unable to get an accuracy above 67% on the validation set. I will need to label more data to improve performance.

## Scraper
When I came up with the idea for this project, my biggest concern was where to find a dataset of journal entries. On a hunch, I searched reddit and found 2 communities with public diary entries (for their privacy I did not name them). Thus, `scraper/scraper.py` contains code to scrape all posts from these communities into a csv which is later parsed into sentences during the labeling phase of the project.

## Labeler
After parsing the posts into sentences, I had 31,818 pieces of data to label, so I needed to create a CLI tool to expedite that process. All code for labeling is contained in `labeler/labeler.py`. My labeling tool contains a few features: 1) Fast labeling 2) Memory of what has been labeled so everything doesn't have to be labeled at once. 3) Randomized choice of sentence to label (I needed this feature to limit my bias). Labeled data is stored in a csv. This part of the project is still in progress.

## Data
This project deals with sensitive data, and thus I have not included it in this repo. If you would like access to this dataset please contact me. If you want to see how I split my data click [here](https://github.com/KyleMoore1/journal.ai/blob/master/data/data-prep.md)

## Train
[see how I trained my model](https://github.com/KyleMoore1/journal.ai/blob/master/train/Train.md)

## Future goals for the project
I hope to achieve an accuracy of at least 70% on my test set. I have a few ideas on how to accomplish this. First, I can get more data. I found another site which contains over 100,000 public journal entries that I can scrape and label. This would reduce the overfitting of my model. Second, I can get rid of the untrue assumption of independence of sentences in a post. I want to try a multimodal approach which both considers the sentence itself as well as the overall post. Finally, I am considering adopting a more structured approach to labeling, with concrete rules for how I label each piece of data. I followed a general heuristic for labeling but I noticed instances where I was labeling certain cases inconsistently which undoubtably affected performance.

## What I learned
This was my first major machine learning project and I learned a lot from it. First, I learned the difficulty of creating and labeling your own dataset. Also, I learned why NLP applications require large datasets and tend to overfit. Finally, I learned various strategies for improving performance of NLP models including transfer learning and language models.
