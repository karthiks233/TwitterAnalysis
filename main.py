import urllib
import urllib.request
from bs4 import BeautifulSoup

from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
import tweepy
import re
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist



li=[]
def search_for_hashtags(consumer_key, consumer_secret, access_token, access_token_secret, hashtag_phrase):
    # create authentication for accessing Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # initialize Tweepy API
    api = tweepy.API(auth,wait_on_rate_limit=True)

    # get the name of the spreadsheet we will write to
    fname = '_'.join(re.findall(r"#(\w+)", hashtag_phrase))

    def clean_tweet(tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        r = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\ / \ / \S+)", " ", tweet).split())
        r = ' '.join(re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \  '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet).split())
        no_punct = ""
        for char in r:
            if char not in punctuations:
                no_punct = no_punct + char
        return no_punct

    def get_tweet_sentiment(tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'


    # open the spreadsheet we will write to
    with open('%s.csv' % (fname), 'w') as file:
        w = csv.writer(file)

        # write header row to spreadsheet
        w.writerow(['tweet_text', 'username', 'all_hashtags', 'followers_count','Location','Sentiment'])
        # w.writerow(['tweet_text', 'username', 'all_hashtags', 'followers_count'])
        # for each tweet matching our hashtags, write relevant info to the spreadsheet
        for tweet in tweepy.Cursor(api.search, q=hashtag_phrase + ' -filter:retweets',
                                   lang="en", tweet_mode='extended').items(100):



             if tweet.user.location == "India" or tweet.user.location == "INDIA":
                 li.append(clean_tweet((tweet.full_text).lower()))
                 w.writerow((tweet.full_text.replace('\n', ' ').encode('utf-8'), tweet.user.screen_name.encode('utf-8'),
                            [e['text'] for e in tweet._json['entities']['hashtags']], tweet.user.followers_count,tweet.user.location,get_tweet_sentiment(tweet.full_text.replace('\n', ' '))))
           





# consumer_key = input('Consumer Key ')
# consumer_secret = input('Consumer Secret ')
# access_token = input('Access Token ')
# access_token_secret = input('Access Token Secret ')
#
# hashtag_phrase = input('Hashtag Phrase ')

search_for_hashtags("uiJXzxgVIsmQL43M3pbId4Lt1", "PHoV0Rcgo8g7WNvqx5xrW0gsPgtSDzjgFzhL5FZ1JVMHpvg6IW", "1183066418444521472-Fer4LX1WGWbdbHPkC5yoZIIsF0EHn2", "oaAxBjvPSNq4JI7Gi6Mq5KGb0qY1NNyJC1o5Tq0yHFnbj", "#yoga")


def noise_removal(noise):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(noise)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


clean_text = []
for i in li:
    clean_text.append(noise_removal(i))

all_words = get_all_words(clean_text)

freq_dist_pos = FreqDist(all_words)


def display(words):
    ## Creating FreqDist for whole BoW, keeping the 20 most common tokens
    all_fdist = FreqDist(words).most_common(10)

    ## Conversion to Pandas series via Python Dictionary for easier plotting
    all_fdist = pd.Series(dict(all_fdist))

    ## Setting figure, ax into variables
    fig, ax = plt.subplots(figsize=(10, 10))

    ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    sns.barplot(x=all_fdist.index, y=all_fdist.values, ax=ax)
    plt.xticks(rotation=30)
    plt.show()
