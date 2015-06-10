__author__ = 'Iosu'

import pandas as pd
import os
import nltk
from static import *
import numpy as np


def print_data(df):
    for index, row in df.iterrows():
        print row['Insult'], row['Comment']

def preprocess_data(df, texts, classes):
    for index, row in df.iterrows():
        texts.append(row['Comment'].decode('utf8'))
        classes.append(row['Insult'])
    return texts, classes

def bow(list_of_words):

    print "Creating the bag of words...\n"
    # from sklearn.feature_extraction.text import CountVectorizer
    #
    # # Initialize the "CountVectorizer" object, which is scikit-learn's
    # # bag of words tool.
    # vectorizer = CountVectorizer(analyzer = "word",   \
    #                              tokenizer = None,    \
    #                              preprocessor = None, \
    #                              stop_words = stopwords,   \
    #                              max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.



    train_data_features = vectorizer.fit_transform(list_of_words)

    # Numpy arrays are easy to work with, so convert the result to an
    # array


    train_data_features = train_data_features.toarray()


    # print train_data_features.shape


    vocab = vectorizer.get_feature_names()
    # print vocab


    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # print dist
    # For each, print the vocabulary word and the number of times it
    # appears in the training set

    dictionary = []
    for tag, count in zip(vocab, dist):
        dictionary.append((tag, count))
        # print count, tag

    dictionary = sorted(dictionary, key=lambda x: x[1])


    return dictionary, train_data_features

def classifier_randomForest():
    print "Training the random forest..."
    from sklearn.ensemble import RandomForestClassifier

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    # print insult_dictionary


    forest = forest.fit(train_data_features_insult, insult_label+no_insult_label)
     # forest = forest.fit(train_data_features_noinsult, no_insult_label)
    return forest


def classifier_svm():
    from sklearn import svm

    clf = svm.LinearSVC()
    clf.fit(train_data_features_insult, insult_label+no_insult_label)
    return clf

def test():

    predictInsult = "you are a fucking asshole idiot"
    predictNormal = "The world is wonderful everything is brilliant"
    predictJeroni = "what in the fuckkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk"


    text=[]
    classes=[]
    test = pd.read_table(os.path.join('Data', 'test_with_solutions.csv'), sep=',', quotechar='"')
    # test = pd.read_table(os.path.join('Data', 'impermium_verification_labels.csv'), sep=',', quotechar='"')
    text, classes = preprocess_data(test, text, classes)

    # text = [predictInsult, predictNormal, predictJeroni]
    # classes = [1, 0, 1]
    text_tokenized=[]
    for idx, texto in enumerate(text):
        text_tokenized.append(ppd.tokenize(texto, classes[idx]))

    list_test=[]
    for t in text_tokenized:
        list_test.append(t['clean'])



    # pred = vectorizer.transform([predictInsult, predictNormal, predictJeroni])
    pred = vectorizer.transform(list_test)
    pred = pred.toarray()
    print pred

    resultRF = forest.predict(pred)
    resultSVM = clf.predict(pred)

    count = 0
    for idx, result in enumerate(resultSVM):
        if result == classes[idx]:
            # print result, classes[idx]
            count = count + 1
        # else:
            # print list_test[idx]

    print count
    print len(resultSVM)
    print 'accuracy svm: '+ str((count*100)/len(resultSVM))
    count = 0
    for idx, result in enumerate(resultRF):
        if result == classes[idx]:
            # print result, classes[idx]
            count = count + 1
        # else:
        #     print list_test[idx]

    print count
    print len(resultRF)
    print 'accuracy rf: '+ str((count*100)/len(resultRF))

    # print resultRF, resultSVM

import preprocessData as ppd

if __name__ == "__main__":

    texts_test = []
    classes_test = []
    df_train = pd.read_table(os.path.join('Data', 'train.csv'), sep=',', quotechar='"')
    df_test_solutions = pd.read_table(os.path.join('Data', 'test_with_solutions.csv'), sep=',', quotechar='"')

    # print_data(df_train)
    # print_data(df_test_solutions)

    text = []
    # texts_test, classes_test = preprocess_data(df_test_solutions, texts_test, classes_test)
    texts_test, classes_test = preprocess_data(df_train, texts_test, classes_test)


    for idx, texto in enumerate(texts_test):
        text.append(ppd.tokenize(texto, classes_test[idx]))


    list_of_insult_words = []
    list_of_words = []
    insult_label = []
    no_insult_label = []

    for t in text:
        if t['class'] == 1:
            list_of_insult_words.append(t['clean'])
            insult_label.append(t['class'])
        else:
            list_of_words.append(t['clean'])
            no_insult_label.append(t['class'])

    # for c in curse_words:
    #     list_of_insult_words.append(c)
    #     insult_label.append(1)


    # print list_of_words

    list_of_words = np.hstack(list_of_words)
    list_of_insult_words = np.hstack(list_of_insult_words)

    # print list_of_words

    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = stopwords,   \
                                 max_features = 5000)

    insult_dictionary, train_data_features_insult = bow(np.concatenate([list_of_insult_words,list_of_words]))
    # no_insult_dictionary, train_data_features_noinsult = bow(list_of_words)

    train_data = []
    # train_data.append()
    print insult_dictionary

    forest = classifier_randomForest()
    clf = classifier_svm()

    test()