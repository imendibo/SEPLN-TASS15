__author__ = 'Iosu'
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier as OvsA
from multilayer_perceptron import MLPClassifier as MLP
from sklearn.lda import LDA
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
import utils as ut

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
    ada = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    ada.fit(features, labels)

    return ada


def train_classifiers(tweets_features, train_labels):

    forest = classifier_randomForest(tweets_features, train_labels)
    svm = classifier_svm(tweets_features, train_labels)
    mlp = multilayer_perceptron(tweets_features, train_labels)
    ada = adaboost(tweets_features, train_labels)
    lr = logistic_regression(tweets_features, train_labels)

    # ONE VS ALL CLASSIFIER WITH DIFFERENT ESTIMATORS.
    estimator = svm.SVC(random_state=0)
    oneVSall_svm = onevsall(tweets_features, train_labels, estimator)
    #
    # estimator = clf.MLP()
    # oneVSall_mlp = clf.onevsall(tweets_features, train_labels, estimator)
    #
    estimator = RandomForestClassifier(n_estimators=50)
    oneVSall_rf = onevsall(tweets_features, train_labels, estimator)


def test_classifiers(pred):
    results, accuracyLR, precisionLR, recallLR, f_measureLR = evaluateResults(lr, pred, test_labels, accuracyLR,
                                                                              precisionLR, recallLR, f_measureLR,
                                                                              estimator_name='Logistic regression')
    results, accuracyRF, precisionRF, recallRF, f_measureRF = evaluateResults(forest, pred, test_labels, accuracyRF,
                                                                              precisionRF, recallRF, f_measureRF,
                                                                              estimator_name='RF')
    results, accuracySVM, precisionSVM, recallSVM, f_measureSVM = evaluateResults(svm, pred, test_labels,
                                                                                  accuracySVM, precisionSVM,
                                                                                  recallSVM, f_measureSVM,
                                                                                  estimator_name='SVM')
    results, accuracyADA, precisionADA, recallADA, f_measureADA = evaluateResults(ada, pred, test_labels,
                                                                                  accuracyADA, precisionADA,
                                                                                  recallADA, f_measureADA,
                                                                                  estimator_name='ADABOOST')
    results, accuracyMLP, precisionMLP, recallMLP, f_measureMLP = evaluateResults(mlp, pred, test_labels,
                                                                                  accuracyMLP, precisionMLP,
                                                                                  recallMLP, f_measureMLP,
                                                                                  estimator_name='MLP')

    results, accuracyOVASVM, precisionOVASVM, recallOVASVM, f_measureOVASVM = evaluateResults(oneVSall_svm, pred,
                                                                                              test_labels,
                                                                                              accuracyOVASVM,
                                                                                              precisionOVASVM,
                                                                                              recallOVASVM,
                                                                                              f_measureOVASVM,
                                                                                              estimator_name='one versus all SVM')
    # evaluateResults(oneVSall_mlp, pred, test_labels, estimator_name='one versus all MLP')
    results, accuracyOVARF, precisionOVARF, recallOVARF, f_measureOVARF = evaluateResults(oneVSall_rf, pred,
                                                                                          test_labels,
                                                                                          accuracyOVARF,
                                                                                          precisionOVARF,
                                                                                          recallOVARF,
                                                                                          f_measureOVARF,
                                                                                          estimator_name='one versus all RF')



def evaluateResults(estimator, test_set, test_labels, accuracy, precision, recall, f_measure, estimator_name='Unknown'):
    result = estimator.predict(test_set)
    # print result

    aux = result == test_labels
    correct = sum(aux.astype(int))
    _accuracy = (correct * 100) / len(test_set)

    # print estimator_name, ': Accuracy = ' + str((correct * 100) / len(test_set)) + "% (" + str(correct) + "/" + str(len(test_set)) + ")"

    cm = ut.get_confusion_matrix(test_labels, result)

    f1_measure = ut.get_f1_measure(test_labels, result)
    _precision, _recall, _f1score, _support = ut.get_measures_for_each_class(test_labels, result)

    results.append(result)
    accuracy.append(_accuracy)
    precision.append(_precision)
    recall.append(_recall)
    f_measure.append(_f1score)

    # print 'Average Precision:\t', _precision
    # print 'Average Recall:\t', _recall
    # print 'Average F1 Measure:\t', _f1score
    # print '\n'

    return results, accuracy, precision, recall, f_measure


