import numpy as np
import os
import classifiers as clf
import BagOfWords as bow
import scipy.stats as st
import utils as ut


def voting(train):
    train_voted = []

    for idx, t in enumerate(train):
        mode = st.mode(t)[0][0]
        train_voted.append(mode)

    return train_voted


def weighted_voting_getlambdas(train, label):
    columns = len(train[0])
    lambdas = []
    normalized_lambdas = []
    results = []
    for i in xrange(0, columns):
        results.append(train[:, i] == label)
    for i in xrange(0, columns):
        lambdas.append(sum(results[i].astype(int)) / float(len(label)))
    for i in xrange(0, len(lambdas)):
        normalized_lambdas.append(lambdas[i] / float(sum(lambdas)))
    return normalized_lambdas


def weighted_voting(test, lambdas):
    voted_result = []

    for idx, t in enumerate(test):
        aux_label = []
        aux_result = []
        for i in xrange(0, len(t)):
            if t[i] in aux_label:
                index = 0
                for idx, al in enumerate(aux_label):
                    if al == t[i]:
                        index = idx
                aux_result[index] += lambdas[i]
            else:
                aux_label.append(t[i])
                aux_result.append(lambdas[i])
        if len(aux_result) == 1:
            voted_result.append(aux_label[0])
        else:
            voted_result.append(aux_label[aux_result.index(max(aux_result))])
    return voted_result


if __name__ == '__main__':
    train_file = 'train_results.csv'
    train_set = np.genfromtxt(train_file, delimiter=',')

    train_label_file = 'train_labels.csv'
    train_label = np.genfromtxt(train_label_file, delimiter=',')

    test_file = 'test_results.csv'
    test_set = np.genfromtxt(test_file, delimiter=',')

    test_label_file = 'test_labels.csv'
    test_label = np.genfromtxt(test_label_file, delimiter=',')

    # columns = len(train_set[0])-1
    # lambdas = [columns]
    # for idx, t in enumerate(train_set):
    # for i in xrange(0, columns):
    #         if t[i] == train_label[idx]:
    #             lambdas[i] += 1
    #     import pdb; pdb.set_trace()


def supperclassify(train_set, train_label, test_set, test_label):
    '''Different methods'''
    train_voted = voting(train_set)
    aux = train_voted == train_label
    correct = sum(aux.astype(int))
    _accuracy = (correct * 100) / len(train_label)
    _precision, _recall, _f1score, _support = ut.get_measures_for_each_class(train_label, train_voted)
    print 'Estimator VOTING'
    print 'Average Accuracy:\t', _accuracy
    print 'Average Precision:\t', _precision
    print 'Average Recall:\t', _recall
    print 'Average F1 Measure:\t', _f1score
    print '\n'

    lambdas = weighted_voting_getlambdas(train_set, train_label)
    results = weighted_voting(test_set, lambdas)

    aux = results == test_label
    correct = sum(aux.astype(int))
    _accuracy = (correct * 100) / len(test_label)
    _precision, _recall, _f1score, _support = ut.get_measures_for_each_class(test_label, results)
    print 'Estimator W_VOTING'
    print 'Average Accuracy:\t', _accuracy
    print 'Average Precision:\t', _precision
    print 'Average Recall:\t', _recall
    print 'Average F1 Measure:\t', _f1score

    rf = clf.classifier_randomForest(train_set, train_label)
    results = clf.evaluateResults(rf, test_set, test_label, estimator_name='RF')

    lr = clf.logistic_regression(train_set, train_label)
    results = clf.evaluateResults(lr, test_set, test_label, estimator_name='LR')

    svm = clf.classifier_svm(train_set, train_label)
    results = clf.evaluateResults(svm, test_set, test_label, estimator_name='SVM')

    rbf = clf.rbf_classifier(train_set, train_label)
    results = clf.evaluateResults(rbf, test_set, test_label, estimator_name='RBF')
    # import pdb; pdb.set_trace()