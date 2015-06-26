__author__ = 'Iosu'

import xmlreader as xml
import utils as ut
import numpy as np
import BagOfWords as bow
import classifiers as clf
import sklearn.cross_validation as cv
import pandas as pd
import matplotlib.pyplot as plt


def printResults(accuracy, precision, recall, f_measure, name="Unknown"):
    print "Result of " + name + ":"
    print "Accuracy: ", sum(accuracy) / float(len(accuracy))
    print "Precision: ", sum(precision) / float(len(precision))
    print "Recall: ", sum(recall) / float(len(recall))
    print "F1-measure: ", sum(f_measure) / float(len(f_measure))


if __name__ == "__main__":

    # xmlTrainFile = '../DATA/general-tweets-train-tagged.xml'
    # tweets = xml.readXML(xmlTrainFile)
    #
    # tokenized_tweets = []
    # for tweet in tweets:
    #     tokenized_tweets.append(ut.tokenize(tweet.content, tweet.polarity))
    #
    # tweets = []
    # labels = []
    # for tweet in tokenized_tweets:
    #     tweets.append(tweet['clean'])
    #     labels.append(tweet['class'])
    #
    # tweets = np.array(tweets)
    # labels = np.array(labels)


    train = pd.read_csv("../Data/imdb/train.tsv", header=0, delimiter="\t", quoting=3)
    # test = pd.read_csv("../Data/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

    tokenized_train = []

    for idx, text in train.iterrows():
        # tokenized_train.append(ut.tokenize(text['review'], text['sentiment'])) # for labeledTrainData.tsv
        tokenized_train.append(ut.tokenize(text['Phrase'], text['Sentiment']))   # for train.tsv

    tweets = []
    labels = []
    for tweet in tokenized_train:
        tweets.append(tweet['clean'])
        labels.append(tweet['class'])

    partition = 3
    # # First Partition
    # train_tweets, test_tweets, validation_tweets, train_labels, test_labels, validation_labels = ut.crossValidation(
    #     tweets, labels, partition)

    # Second partition
    train_tweets, validation_tweets, test_tweets, train_labels, validation_labels, test_labels = ut.crossValidation(
        tweets, labels, partition)

    print len(test_tweets)
    print len(train_tweets)
    print len(validation_tweets)

    train_tweets = np.hstack(train_tweets)
    dictionary, tweets_features, vectorizer = bow.bow(train_tweets, vec="tfidf")
    # dictionary, tweets_features, vectorizer = bow.bow(train_tweets, vec="count")


    '''
    Training different classifiers.
    '''
    print '\nTraining Classifiers:\n'
    # forest_cls, svm_cls, rbf_cls, ada_cls, lr_cls = clf.train_classifiers(tweets_features,train_labels)
    forest_cls, svm_cls, lr_cls, ada_cls = clf.train_classifiers(tweets_features,train_labels)
    '''
    Create results dataset from classifiers. Where each attribute is a classifier and each row corresponds to the
    classification of a tweet according to each classifier.

    '''
    print '\nCreating Train set for super classifier ... '
    test_tweet_trans = vectorizer.transform(test_tweets)
    test_tweet_trans = test_tweet_trans.toarray()

    # classifiers = (forest_cls, svm_cls, rbf_cls, ada_cls, lr_cls)
    classifiers = (forest_cls, svm_cls, lr_cls, ada_cls)
    train_results = clf.test_classifiers(test_tweet_trans, test_labels, classifiers)

    '''
    Train the super classifier on the test set
    '''
    print '\nCreating Test set for super classifier ... '
    val_tweet_trans = vectorizer.transform(validation_tweets)
    val_tweet_trans = val_tweet_trans.toarray()

    test_results = clf.test_classifiers(val_tweet_trans, validation_labels, classifiers)

    '''
    Now we have a train_results and test_results. Lets train and test a super classifier
    '''
    print '\nTraining super classifier ... '
    super_clf = clf.rbf_classifier(train_results, test_labels)

    print '\nEvaluating Super classifier ... '
    results, accuracy, precision, recall, f_measure = clf.evaluateResults(super_clf, test_results, validation_labels, estimator_name='Supper Classifier')
    import pdb; pdb.set_trace()
    np.savetxt("train_results_2.csv", train_results, delimiter=",")
    np.savetxt("train_labels_2.csv", test_labels, delimiter=",")
    np.savetxt("test_results_2.csv", test_results, delimiter=",")
    np.savetxt("test_labels_2.csv", validation_labels, delimiter=",")

    printResults(accuracy, precision, recall, f_measure, name="SUPER CLASSIFICATOR")
