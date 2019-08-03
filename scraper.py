import praw
import csv

reddit = praw.Reddit(client_id='e8Xh2lBjUriwkw', client_secret='Ci58j2njJpsccTushvKefBoYrTU', user_agent='diary_scrape')


# get all posts from subreddit
submissions = reddit.subreddit('Diary').new(limit = 10000)
for submission in submissions:
    textEntry = (submission.title + "\n" + submission.selftext)
    row = [0, textEntry]
    with open('data.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
