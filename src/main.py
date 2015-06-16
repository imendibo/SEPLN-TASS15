__author__ = 'Iosu'

import xmlreader as xml
import utils as ut
import numpy as np
import BagOfWords as bow
import classifiers as clf


def test(forest, svm, oneVSall, vectorizer, test_tweets, test_labels):
    pred = vectorizer.transform(test_tweets)
    pred = pred.toarray()
    print pred

    resultSVM = svm.predict(pred)
    resultRF = forest.predict(pred)
    resultOvsA = oneVSall.predict(pred)

    count = 0
    for idx, result in enumerate(resultSVM):
        if result == test_labels[idx]:
            count += 1

    print count
    print len(resultSVM)
    print 'accuracy svm: ' + str((count * 100) / len(resultSVM))

    count = 0
    for idx, result in enumerate(resultRF):
        if result == test_labels[idx]:
            count += 1

    print count
    print len(resultRF)
    print 'accuracy rf: ' + str((count * 100) / len(resultRF))

    count = 0
    for idx, result in enumerate(resultOvsA):
        if result == test_labels[idx]:
            count += 1

    print count
    print len(resultOvsA)
    print 'accuracy one vs all: ' + str((count * 100) / len(resultOvsA))


if __name__ == "__main__":

    xmlTrainFile = '../DATA/general-tweets-train-tagged.xml'
    tweets = xml.readXML(xmlTrainFile)

    tokenized_tweets = []
    test_labels = []
    train_labels = []
    for tweet in tweets:
        tokenized_tweets.append(ut.tokenize(tweet.content, tweet.polarity))

    count = 0
    test_tweets = []
    train_tweets = []

    # total_tweets = []
    # total_labels = []
    #
    # for tweet in tokenized_tweets:
    # total_tweets.append(tweet['clean'])
    # total_labels.append(tweet['class'])


    partition = 5
    train_tweets, train_labels, test_tweets, test_labels = ut.partition_data(tokenized_tweets, partition)

    print len(test_tweets)
    print len(train_tweets)

    # for t in train_tweets:
    # print t.encode('ascii', 'replace')
    # clean_tweets.append(tweet['clean'])
    # labels.append(tweet['class'])

    train_tweets = np.hstack(train_tweets)
    dictionary, tweets_features, vectorizer = bow.bow(train_tweets)

    # print dictionary

    forest = clf.classifier_randomForest(tweets_features, train_labels)
    svm = clf.classifier_svm(tweets_features, train_labels)
    oneVSall = clf.onevsall(tweets_features, train_labels)

    test(forest, svm, oneVSall, vectorizer, test_tweets, test_labels)