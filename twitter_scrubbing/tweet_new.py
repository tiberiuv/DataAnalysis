import csv
import tweepy
from textblob import TextBlob
import json

with open('twitter_credentials.json') as cred_data:
    info = json.load(cred_data)
    consumer_key = info['CONSUMER_KEY']
    consumer_secret = info['CONSUMER_SECRET']
    access_key = info['ACCESS_KEY']
    access_secret = info['ACCESS_SECRET']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

# passing the auth to tweepy API which provide gateway to tweets
api = tweepy.API(auth)

# opening a csv file
csvFile = open('result.csv', 'a')
csvWriter = csv.writer(csvFile)

# receiving keyword you want to search for
public_tweets = api.search(input("Topic you want to analyse for: "))

# running a for loop to iterate over tweets and printing one row at a time
for tweet in public_tweets:
    # write in a csv file
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
