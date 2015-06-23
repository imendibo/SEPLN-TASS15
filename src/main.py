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


    return results, accuracy, _precision, _recall, _f1score, _support


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

    partition = 5
    # train_tweets, test_tweets, train_labels, test_labels = ut.crossValidation(tweets, labels, partition)

    kf = cv.KFold(n=len(tweets), n_folds=3, shuffle=True, indices=False)

    accuracyLR, precisionLR, recallLR, f_measureLR = [], [], [], []
    accuracyRF, precisionRF, recallRF, f_measureRF = [], [], [], []
    accuracySVM, precisionSVM, recallSVM, f_measureSVM = [], [], [], []
    accuracyADA, precisionADA, recallADA, f_measureADA = [], [], [], []
    accuracyMLP, precisionMLP, recallMLP, f_measureMLP = [], [], [], []
    accuracyOVASVM, precisionOVASVM, recallOVASVM, f_measureOVASVM = [], [], [], []
    accuracyOVARF, precisionOVARF, recallOVARF, f_measureOVARF = [], [], [], []
    results = []

    for train, test in kf:
        # print "Fold "
        train = np.array(train)
        test = np.array(test)
        tweets = np.array(tweets)
        labels = np.array(labels)

        # for train, test in kf:
        # train_tweets = tweets[np.array(train)]
        train_tweets, test_tweets, train_labels, test_labels = tweets[train], tweets[test], labels[train], labels[test]
        # train_tweets, train_labels, test_tweets, test_labels = ut.partition_data(tokenized_tweets, partition)

        # print len(test_tweets)
        # print len(train_tweets)

        train_tweets = np.hstack(train_tweets)
        dictionary, tweets_features, vectorizer = bow.bow(train_tweets, vec="tfidf")
        # dictionary, tweets_features, vectorizer = bow.bow(train_tweets, vec="count")

        # print dictionary

        '''Dimsionality reduction'''
        # LDA
        # lda = clf.lda(tweets_features, train_labels)
        # print tweets_features.shape

        '''
        Training different classifiers.
        '''
        forest = clf.classifier_randomForest(tweets_features, train_labels)
        svm = clf.classifier_svm(tweets_features, train_labels)
        mlp = clf.multilayer_perceptron(tweets_features, train_labels)
        ada = clf.adaboost(tweets_features, train_labels)
        lr = clf.logistic_regression(tweets_features, train_labels)

        # ONE VS ALL CLASSIFIER WITH DIFFERENT ESTIMATORS.
        estimator = clf.svm.SVC(random_state=0)
        oneVSall_svm = clf.onevsall(tweets_features, train_labels, estimator)
        #
        # estimator = clf.MLP()
        # oneVSall_mlp = clf.onevsall(tweets_features, train_labels, estimator)
        #
        estimator = clf.RandomForestClassifier(n_estimators=50)
        oneVSall_rf = clf.onevsall(tweets_features, train_labels, estimator)

        '''
        Test the different classifiers with the test tweets.
        '''

        pred = vectorizer.transform(test_tweets)
        pred = pred.toarray()
        # pred = SelectKBest(chi2, k=4500).fit_transform(pred, test_labels)


        # evaluateResults(lda, pred, test_labels, estimator_name='LDA')


        results, accuracyLR, precisionLR, recallLR, f_measureLR = clf.evaluateResults(lr, pred, test_labels,
                                                                                  estimator_name='Logistic regression')
        results, accuracyRF, precisionRF, recallRF, f_measureRF = clf.evaluateResults(forest, pred, test_labels,
                                                                                  estimator_name='RF')
        results, accuracySVM, precisionSVM, recallSVM, f_measureSVM = clf.evaluateResults(svm, pred, test_labels,
                                                                                      estimator_name='SVM')
        results, accuracyADA, precisionADA, recallADA, f_measureADA = clf.evaluateResults(ada, pred, test_labels,
                                                                                      estimator_name='ADABOOST')
        results, accuracyMLP, precisionMLP, recallMLP, f_measureMLP = clf.evaluateResults(mlp, pred, test_labels,
                                                                                      estimator_name='MLP')

        results, accuracyOVASVM, precisionOVASVM, recallOVASVM, f_measureOVASVM = clf.evaluateResults(oneVSall_svm, pred,
                                                                                                  test_labels,
                                                                                                  estimator_name='one versus all SVM')
        # evaluateResults(oneVSall_mlp, pred, test_labels, estimator_name='one versus all MLP')
        results, accuracyOVARF, precisionOVARF, recallOVARF, f_measureOVARF = clf.evaluateResults(oneVSall_rf, pred,
                                                                                              test_labels,
                                                                                              estimator_name='one versus all RF')

    printResults(accuracyLR, precisionLR, recallLR, f_measureLR, name="LR")
    printResults(accuracyRF, precisionRF, recallRF, f_measureRF, name="RF")
    printResults(accuracySVM, precisionSVM, recallSVM, f_measureSVM, name="SVM")
    printResults(accuracyADA, precisionADA, recallADA, f_measureADA, name="ADABOOST")
    printResults(accuracyMLP, precisionMLP, recallMLP, f_measureMLP, name="MLP")
    printResults(accuracyOVASVM, precisionOVASVM, recallOVASVM, f_measureOVASVM, name="OVA SVM")
    printResults(accuracyOVARF, precisionOVARF, recallOVARF, f_measureOVARF, name="OVA RF")

