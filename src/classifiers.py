__author__ = 'Iosu'
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier as OvsA
from multilayer_perceptron import MLPClassifier as MLP
from sklearn.lda import LDA
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier

def classifier_randomForest(features, labels):
    print "Training the random forest..."

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