__author__ = 'Iosu'

import xml.etree.ElementTree as ET
import Tweet as tw


def readXML(xmlFIle):
    tree = ET.parse(xmlFIle)
    root = tree.getroot()

    tweets = []

    for tweet in root.iter('tweet'):
        content = tweet.find('content').text

        sentiments = tweet.find('sentiments')
        polarity = sentiments[0].find('value').text


        #Other info:
        tweet_id = long(tweet.find('tweetid').text)
        user = tweet.find('user').text
        date = tweet.find('date').text
        lang = tweet.find('lang').text

        tweet = tw.Tweet(tweet_id, user, date, lang, content, polarity)

        tweets.append(tweet)

    return tweets