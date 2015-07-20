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
from sklearn.preprocessing import normalize

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
    print '\n\tTraining Random Forest..'
    forest_cls = classifier_randomForest(tweets_features, train_labels)
    print '\n\tTraining Linear SVM..'
    svm_cls = classifier_svm(tweets_features, train_labels)
    print '\n\tTraining Linear Regression..'
    lr_cls = logistic_regression(tweets_features, train_labels)
    # print '\n\tTraining RBF SVM..'
    # rbf_cls = rbf_classifier(tweets_features,train_labels)
    print '\n\tTraining Adaboost..'
    ada_cls = adaboost(tweets_features, train_labels)

    # ONE VS ALL CLASSIFIER WITH DIFFERENT ESTIMATORS.
    # estimator = svm.SVC(random_state=0)
    # oneVSall_svm_cls = onevsall(tweets_features, train_labels, estimator)
    #
    # estimator = clf.MLP()
    # oneVSall_mlp = clf.onevsall(tweets_features, train_labels, estimator)
    #
    # estimator = RandomForestClassifier(n_estimators=50)
    # oneVSall_rf_cls = onevsall(tweets_features, train_labels, estimator)
    # return forest_cls, svm_cls, rbf_cls, ada_cls, lr_cls
    return forest_cls, svm_cls, lr_cls, ada_cls
#
# def test_classifiers(pred, test_labels, classifiers):
#     forest_cls, svm_cls, rbf_cls, ada_cls, lr_cls = classifiers
#     results_lr = get_classification(lr_cls, pred, test_labels, estimator_name='Logistic regression')
#     results_forest = get_classification(forest_cls, pred, test_labels, estimator_name='RF')
#     results_svm = get_classification(svm_cls, pred, test_labels, estimator_name='SVM')
#     results_ada = get_classification(ada_cls, pred, test_labels, estimator_name='ADABOOST')
#     results_rbf = get_classification(rbf_cls, pred, test_labels, estimator_name='RBF')
#     results = np.column_stack([results_lr, results_forest, results_svm, results_ada, results_rbf])
#     return results


def test_classifiers(pred, test_labels, classifiers):
    forest_cls, svm_cls, lr_cls, ada_cls = classifiers
    results_forest = get_classification(forest_cls, pred, test_labels, estimator_name='RF')
    results_svm = get_classification(svm_cls, pred, test_labels, estimator_name='SVM')
    results_lr = get_classification(lr_cls, pred, test_labels, estimator_name='Logistic regression')
    results_ada = get_classification(ada_cls, pred, test_labels, estimator_name='ADABOOST')
    results = np.column_stack([results_forest, results_svm, results_lr, results_ada])
    return results


def evaluateResults(estimator, test_set, test_labels, estimator_name='Unknown', file_name=''):
    result = estimator.predict(test_set)
    # print result

    aux = result == test_labels
    correct = sum(aux.astype(int))
    _accuracy = (correct * 100) / len(test_set)

    cm = ut.get_confusion_matrix(test_labels, result, estimator_name, file_name=file_name)
    print '\n'
    _precision, _recall, _f1score, _support = ut.get_measures_for_each_class(test_labels, result)
    print 'Estimator ', estimator_name
    print 'Average Accuracy:\t', _accuracy
    print 'Average Precision:\t', _precision
    print 'Average Recall:\t', _recall
    print 'Average F1 Measure:\t', _f1score
    print '\n'
    return result, _accuracy, _precision, _recall, _f1score


def get_classification(estimator, test_set, test_labels, estimator_name='Unknown'):
    result = estimator.predict(test_set)
    return result
