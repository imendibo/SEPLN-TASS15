__author__ = 'Iosu'


import xmlreader as xml
import utils as ut
import numpy as np
import BagOfWords as bow

if __name__ == "__main__":

    xmlTrainFile = '../DATA/general-tweets-train-tagged.xml'
    tweets = xml.readXML(xmlTrainFile)

    tokenized_tweets = []
    labels = []

    for tweet in tweets:
        tokenized_tweets.append(ut.tokenize(tweet.content, tweet.polarity))
        # labels.append(tweet.polarity)

    clean_tweets = []
    for tweet in tokenized_tweets:
        clean_tweets.append(tweet['clean'])
        labels.append(tweet['class'])

    clean_tweets = np.hstack(clean_tweets)

    dictionary, tweets_features, vectorizer = bow.bow(clean_tweets)

    
