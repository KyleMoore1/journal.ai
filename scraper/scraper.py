import praw
import csv

reddit = praw.Reddit(client_id='<your client id here>', client_secret='<your client secret here>', user_agent='<your user agent here>')


# get all posts from subreddit
submissions = reddit.subreddit('<your community name here>').new(limit = 10000)
for submission in submissions:
    textEntry = (submission.selftext)
    row = [textEntry]
    with open('../data/data.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
