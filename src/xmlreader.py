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

        # polarity = polarityTagging(polarity)
        polarity = polarityTagging3(polarity)

        # Other info:
        tweet_id = long(tweet.find('tweetid').text)
        user = tweet.find('user').text
        date = tweet.find('date').text
        lang = tweet.find('lang').text

        if content != None:
            tweet = tw.Tweet(tweet_id, user, date, lang, content, polarity)

            tweets.append(tweet)

    return tweets


def readXMLTest(xmlFIle):
    tree = ET.parse(xmlFIle)
    root = tree.getroot()

    tweets = []

    for tweet in root.iter('tweet'):
        content = tweet.find('content').text

        # sentiments = tweet.find('sentiments')
        # polarity = sentiments[0].find('value').text
        polatity = 'NONE'
        # polarity = polarityTagging(polarity)

        # Other info:
        tweet_id = long(tweet.find('tweetid').text)
        user = tweet.find('user').text
        date = tweet.find('date').text
        lang = tweet.find('lang').text

        if content != None:
            tweet = tw.Tweet(tweet_id, user, date, lang, content, polatity)

            tweets.append(tweet)

    return tweets


def polarityTagging(polarity):
    if (polarity.__eq__('NONE')):
        polarity = 0
    elif (polarity.__eq__('N+')):
        polarity = 1
    elif (polarity.__eq__('N')):
        polarity = 2
    elif (polarity.__eq__('NEU')):
        polarity = 3
    elif (polarity.__eq__('P')):
        polarity = 4
    elif (polarity.__eq__('P+')):
        polarity = 5

    return polarity


def polarityTagging3(polarity):
    if (polarity.__eq__('NONE')):
        polarity = 0
    elif (polarity.__eq__('N+')):
        polarity = 1
    elif (polarity.__eq__('N')):
        polarity = 1
    elif (polarity.__eq__('NEU')):
        polarity = 2
    elif (polarity.__eq__('P')):
        polarity = 3
    elif (polarity.__eq__('P+')):
        polarity = 3

    return polarity