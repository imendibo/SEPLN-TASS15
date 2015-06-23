__author__ = 'Iosu'
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier as OvsA
from multilayer_perceptron import MLPClassifier as MLP
from sklearn.lda import LDA
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
import utils as ut
import numpy as np


def classifier_randomForest(features, labels):
    # print "Training the random forest..."
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    forest = forest.fit(features, labels)
    return forest


def classifier_svm(features, labels):
    clf_svm = svm.LinearSVC()
    clf_svm.fit(features, labels)
    return clf_svm


def onevsall(tweets_features, train_labels, estimator):
    clf_ova = OvsA(estimator)
    clf_ova.fit(tweets_features, train_labels)
    return clf_ova

def rbf_classifier(features, labels):
    clf_rbf = svm.SVC(random_state=0)
    clf_rbf.fit(features, labels)
    return clf_rbf

def multilayer_perceptron(tweet_features, train_labels):
    clf_mlp = MLP(n_hidden=100)
    clf_mlp.fit(tweet_features, train_labels)
    return clf_mlp


def lda(tweet_features, train_labels):
    clf = LDA()
    clf.fit(tweet_features, train_labels)
    return clf


def logistic_regression(tweet_features, train_labels):
    logreg = linear_model.LogisticRegression()
    logreg.fit(tweet_features, train_labels)
    return logreg


def adaboost(features, labels):
    ada = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R',
                             random_state=None)
    ada.fit(features, labels)

    return ada


def train_classifiers(tweets_features, train_labels):
    forest_cls = classifier_randomForest(tweets_features, train_labels)
    svm_cls = classifier_svm(tweets_features, train_labels)
    mlp_cls = multilayer_perceptron(tweets_features, train_labels)
    ada_cls = adaboost(tweets_features, train_labels)
    lr_cls = logistic_regression(tweets_features, train_labels)

    # ONE VS ALL CLASSIFIER WITH DIFFERENT ESTIMATORS.
    estimator = svm.SVC(random_state=0)
    oneVSall_svm_cls = onevsall(tweets_features, train_labels, estimator)
    #
    # estimator = clf.MLP()
    # oneVSall_mlp = clf.onevsall(tweets_features, train_labels, estimator)
    #
    estimator = RandomForestClassifier(n_estimators=50)
    oneVSall_rf_cls = onevsall(tweets_features, train_labels, estimator)
    return forest_cls, svm_cls, mlp_cls, ada_cls, lr_cls, oneVSall_svm_cls, oneVSall_rf_cls


def test_classifiers(pred, test_labels, classifiers):
    forest_cls, svm_cls, mlp_cls, ada_cls, lr_cls, ova_svm_cls, ova_rf_cls = classifiers
    results_lr = get_classification(lr_cls, pred, test_labels, estimator_name='Logistic regression')
    results_forest = get_classification(forest_cls, pred, test_labels, estimator_name='RF')
    results_svm = get_classification(svm_cls, pred, test_labels, estimator_name='SVM')
    results_ada = get_classification(ada_cls, pred, test_labels, estimator_name='ADABOOST')
    results_mlp = get_classification(mlp_cls, pred, test_labels, estimator_name='MLP')
    results_ova_svm = get_classification(ova_svm_cls, pred, test_labels, estimator_name='one versus all SVM')
    results_ova_rf = get_classification(ova_rf_cls, pred, test_labels, estimator_name='one versus all RF')

    results = np.column_stack([results_lr, results_forest, results_svm, results_ada, results_mlp,
                               results_ova_svm, results_ova_rf])
    return results


def evaluateResults(estimator, test_set, test_labels, estimator_name='Unknown'):
    result = estimator.predict(test_set)
    # print result

    aux = result == test_labels
    correct = sum(aux.astype(int))
    _accuracy = (correct * 100) / len(test_set)

    cm = ut.get_confusion_matrix(test_labels, result)
    _precision, _recall, _f1score, _support = ut.get_measures_for_each_class(test_labels, result)

    print 'Average Precision:\t', _precision
    print 'Average Recall:\t', _recall
    print 'Average F1 Measure:\t', _f1score
    print '\n'

    return result, _accuracy, _precision, _recall, _f1score


def get_classification(estimator, test_set, test_labels, estimator_name='Unknown'):
    result = estimator.predict(test_set)
    return result
