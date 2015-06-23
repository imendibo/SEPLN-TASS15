__author__ = 'Iosu'

import xmlreader as xml
import utils as ut
import numpy as np
import BagOfWords as bow
import classifiers as clf
import sklearn.cross_validation as cv

import matplotlib.pyplot as plt


def printResults(accuracy, precision, recall, f_measure, name="Unknown"):
    print "Result of " + name + ":"
    print "Accuracy: ", sum(accuracy) / float(len(accuracy))
    print "Precision: ", sum(precision) / float(len(precision))
    print "Recall: ", sum(recall) / float(len(recall))
    print "F1-measure: ", sum(f_measure) / float(len(f_measure))


if __name__ == "__main__":

    xmlTrainFile = '../DATA/general-tweets-train-tagged.xml'
    tweets = xml.readXML(xmlTrainFile)

    tokenized_tweets = []
    for tweet in tweets:
        tokenized_tweets.append(ut.tokenize(tweet.content, tweet.polarity))

    tweets = []
    labels = []
    for tweet in tokenized_tweets:
        tweets.append(tweet['clean'])
        labels.append(tweet['class'])

    tweets = np.array(tweets)
    labels = np.array(labels)

    partition = 3
    train_tweets, test_tweets, validation_tweets, train_labels, test_labels, validation_labels = ut.crossValidation(
        tweets, labels, partition)

    print len(test_tweets)
    print len(train_tweets)
    print len(validation_tweets)

    train_tweets = np.hstack(train_tweets)
    dictionary, tweets_features, vectorizer = bow.bow(train_tweets, vec="tfidf")
    # dictionary, tweets_features, vectorizer = bow.bow(train_tweets, vec="count")

    # print dictionary

    '''
    Training different classifiers.
    '''
    forest_cls, svm_cls, mlp_cls, ada_cls, lr_cls, ova_svm_cls, ova_rf_cls = clf.train_classifiers(tweets_features,
                                                                                                   train_labels)

    '''
    Test the different classifiers with the test tweets.
    '''


    test_tweet_trans= vectorizer.transform(test_tweets)
    test_tweet_trans = test_tweet_trans.toarray()

    classifiers = (forest_cls, svm_cls, mlp_cls, ada_cls, lr_cls, ova_svm_cls, ova_rf_cls)
    results = clf.test_classifiers(test_tweet_trans, test_labels, classifiers)

    super_cl = clf.classifier_svm(results, test_labels)


    pred = vectorizer.transform(validation_tweets)
    pred = pred.toarray()


    printResults(accuracyLR, precisionLR, recallLR, f_measureLR, name="LR")
    printResults(accuracyRF, precisionRF, recallRF, f_measureRF, name="RF")
    printResults(accuracySVM, precisionSVM, recallSVM, f_measureSVM, name="SVM")
    printResults(accuracyADA, precisionADA, recallADA, f_measureADA, name="ADABOOST")
    printResults(accuracyMLP, precisionMLP, recallMLP, f_measureMLP, name="MLP")
    printResults(accuracyOVASVM, precisionOVASVM, recallOVASVM, f_measureOVASVM, name="OVA SVM")
    printResults(accuracyOVARF, precisionOVARF, recallOVARF, f_measureOVARF, name="OVA RF")
