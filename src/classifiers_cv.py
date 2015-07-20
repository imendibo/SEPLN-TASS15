__author__ = 'Iosu'

import os

import numpy as np
import sklearn.cross_validation as cv

import xmlreader as xml
import utils as ut
import BagOfWords as bow
import classifiers as clf


def printResults(accuracy, precision, recall, f_measure, name="Unknown"):
    print '\n'
    print "Result of " + name + ":"
    print "Accuracy: ", sum(accuracy) / float(len(accuracy))
    print "Precision: ", sum(precision) / float(len(precision))
    print "Recall: ", sum(recall) / float(len(recall))
    print "F1-measure: ", sum(f_measure) / float(len(f_measure))
    print '\n'


if __name__ == "__main__":

    xmlTrainFile = '../DATA/general-tweets-train-tagged.xml'
    tweets = xml.readXML(xmlTrainFile)

    results_folder = 'results_5/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    tokenized_tweets = []
    for tweet in tweets:
        tokenized_tweets.append(ut.tokenize(tweet.content, tweet.polarity))

    tweets = []
    labels = []
    for tweet in tokenized_tweets:
        tweets.append(tweet['clean'])
        labels.append(tweet['class'])

    kf = cv.KFold(n=len(tweets), n_folds=3, shuffle=True, indices=False)
    accuracyLR, precisionLR, recallLR, f_measureLR = [], [], [], []
    accuracyRF, precisionRF, recallRF, f_measureRF = [], [], [], []
    accuracySVM, precisionSVM, recallSVM, f_measureSVM = [], [], [], []
    accuracyADA, precisionADA, recallADA, f_measureADA = [], [], [], []
    results = []
    for train, test in kf:
        print 'Fold\n'

        train = np.array(train)
        test = np.array(test)

        # Get the whole tweet datset
        tweets = np.array(tweets)
        labels = np.array(labels)

        # Extract tweet partition
        train_tweets = tweets[np.array(train)]
        train_tweets, test_tweets, train_labels, test_labels = tweets[train], tweets[test], labels[train], labels[test]

        print len(test_tweets)
        print len(train_tweets)

        train_tweets = np.hstack(train_tweets)
        dictionary, tweets_features, vectorizer = bow.bow(train_tweets, vec="tfidf")
        '''
        Training different classifiers.
        '''
        svm = clf.classifier_svm(tweets_features, train_labels)
        rf = clf.classifier_randomForest(tweets_features, train_labels)
        ada = clf.adaboost(tweets_features, train_labels)
        lr = clf.logistic_regression(tweets_features, train_labels)

        '''
        Test the different classifiers with the test tweets.
        '''

        pred = vectorizer.transform(test_tweets)
        pred = pred.toarray()

        _results, _accuracyLR, _precisionLR, _recallLR, _f_measureLR = clf.evaluateResults(lr, pred, test_labels,
                                                                                           estimator_name='Logistic regression',
                                                                                           file_name=results_folder)
        _results, _accuracyRF, _precisionRF, _recallRF, _f_measureRF = clf.evaluateResults(rf, pred, test_labels,
                                                                                           estimator_name='RF',
                                                                                           file_name=results_folder)

        _results, _accuracySVM, _precisionSVM, _recallSVM, _f_measureSVM = clf.evaluateResults(svm, pred, test_labels,
                                                                                               estimator_name='SVM',
                                                                                               file_name=results_folder)
        _results, _accuracyADA, _precisionADA, _recallADA, _f_measureADA = clf.evaluateResults(ada, pred, test_labels,
                                                                                               estimator_name='ADABOOST',
                                                                                               file_name=results_folder)

        accuracyLR.append(_accuracyLR)
        precisionLR.append(_precisionLR)
        recallLR.append(_recallLR)
        f_measureLR.append(_f_measureLR)

        accuracyRF.append(_accuracyRF)
        precisionRF.append(_precisionRF)
        recallRF.append(_recallRF)
        f_measureRF.append(_f_measureRF)

        accuracySVM.append(_accuracySVM)
        precisionSVM.append(_precisionSVM)
        recallSVM.append(_recallSVM)
        f_measureSVM.append(_f_measureSVM)

        accuracyADA.append(_accuracyADA)
        precisionADA.append(_precisionADA)
        recallADA.append(_recallADA)
        f_measureADA.append(_f_measureADA)

    printResults(accuracyRF, precisionRF, recallRF, f_measureRF, name="Average RF")
    printResults(accuracyLR, precisionLR, recallLR, f_measureLR, name="Average LR")
    printResults(accuracySVM, precisionSVM, recallSVM, f_measureSVM, name="Average SVM")
    printResults(accuracyADA, precisionADA, recallADA, f_measureADA, name="Average ADA")