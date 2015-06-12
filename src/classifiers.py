__author__ = 'Iosu'
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier as OvsA

def classifier_randomForest(features, labels):
    print "Training the random forest..."

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=50)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    forest = forest.fit(features, labels)
    return forest

def classifier_svm(features, labels):
    clf_svm = svm.LinearSVC()
    clf_svm.fit(features, labels)
    return clf_svm

def onevsall(tweets_features, train_labels):
    clf_ova = OvsA(svm.SVC(random_state=0))
    clf_ova.fit(tweets_features, train_labels)
    return clf_ova

    #Iosu's shit
    # ova = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(tweets_features, train_labels)
    # return ova
