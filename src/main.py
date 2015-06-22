__author__ = 'Iosu'

import xmlreader as xml
import utils as ut
import numpy as np
import BagOfWords as bow
import classifiers as clf
import sklearn.cross_validation as cv

import matplotlib.pyplot as plt


def evaluateResults(estimator, test_set, test_labels, estimator_name='Unknown'):

    result = estimator.predict(test_set)
    print result

    aux = result == test_labels
    correct = sum(aux.astype(int))

    print estimator_name, ': Accuracy = ' + str((correct * 100) / len(test_set)) + "% (" + str(correct) + "/" + str(
        len(test_set)) + ")"

    cm = ut.get_confusion_matrix(test_labels, result)

    f1_measure = ut.get_f1_measure(test_labels, result)
    _precision, _recall, _f1score, _support = ut.get_measures_for_each_class(test_labels, result)

    print 'Average Precision:\t', _precision
    print 'Average Recall:\t', _recall
    print 'Average F1 Measure:\t', _f1score
    print '\n'



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

    for train, test in kf:
        print "Fold "
        train = np.array(train)
        test = np.array(test)
        tweets = np.array(tweets)
        labels = np.array(labels)

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
        print tweets_features.shape

        '''
        Training different classifiers.
        '''
        # forest = clf.classifier_randomForest(tweets_features, train_labels)
        # svm = clf.classifier_svm(tweets_features, train_labels)
        # mlp = clf.multilayer_perceptron(tweets_features, train_labels)
        ada = clf.adaboost(tweets_features, train_labels)

        # lr = clf.logistic_regression(tweets_features, train_labels)

        # ONE VS ALL CLASSIFIER WITH DIFFERENT ESTIMATORS.
        # estimator = clf.svm.SVC(random_state=0)
        # oneVSall_svm = clf.onevsall(tweets_features, train_labels, estimator)
        #
        # estimator = clf.MLP()
        # oneVSall_mlp = clf.onevsall(tweets_features, train_labels, estimator)
        #
        # estimator = clf.RandomForestClassifier(n_estimators=50)
        # oneVSall_rf = clf.onevsall(tweets_features, train_labels, estimator)

        '''
        Test the different classifiers with the test tweets.
        '''

        pred = vectorizer.transform(test_tweets)
        pred = pred.toarray()

        # pred = SelectKBest(chi2, k=4500).fit_transform(pred, test_labels)

        print pred

        # evaluateResults(lda, pred, test_labels, estimator_name='LDA')
        # evaluateResults(lr, pred, test_labels, estimator_name='Logistic regression')

        # evaluateResults(forest, pred, test_labels, estimator_name='RF')
        # evaluateResults(svm, pred, test_labels, estimator_name='SVM')
        evaluateResults(ada, pred, test_labels, estimator_name='ADABOOST')


        # evaluateResults(mlp, pred, test_labels, estimator_name='MLP')
        # evaluateResults(oneVSall_svm, pred, test_labels, estimator_name='one versus all SVM')
        # evaluateResults(oneVSall_mlp, pred, test_labels, estimator_name='one versus all MLP')
        # evaluateResults(oneVSall_rf, pred, test_labels, estimator_name='one versus all RF')
