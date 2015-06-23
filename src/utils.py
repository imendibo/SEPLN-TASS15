__author__ = 'Iosu'
import re
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import sklearn.cross_validation as cv


def tokenize(original_text, label):
    caps = sum([1 for ch in original_text if 'A' <= ch <= 'Z'])
    if caps:
        total = caps + sum([1 for ch in original_text if 'a' <= ch <= 'z'])
        ratio = float(caps) / total
    else:
        ratio = 0

    text = original_text.lower()
    text = text[0:-1]

    # Encodings....
    text = re.sub(r'\\\\', r'\\', text)
    text = re.sub(r'\\\\', r'\\', text)
    text = re.sub(r'\\x\w{2,2}', ' ', text)
    text = re.sub(r'\\u\w{4,4}', ' ', text)
    text = re.sub(r'\\n', ' . ', text)

    # Remove email adresses
    text = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', ' ', text)
    # Remove urls
    text = re.sub(r'\w+:\/\/\S+', r' ', text)
    # Format whitespaces
    text = text.replace('"', ' ')
    text = text.replace('\'', ' ')
    text = text.replace('_', ' ')
    text = text.replace('-', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub(' +', ' ', text)

    # clean_text = text
    text = text.replace('\'', ' ')
    text = re.sub(' +', ' ', text)

    # Is somebody cursing?
    text = re.sub(r'([#%&\*\$]{2,})(\w*)', r'\1\2 ', text)
    # Remove repeated question marks
    text = re.sub(r'([^!\?])(\?{2,})(\Z|[^!\?])', r'\1 \n\3', text)
    # Remove repeated question marks
    text = re.sub(r'([^\.])(\.{2,})', r'\1 \n', text)
    # Remove repeated exclamation (and also question) marks
    text = re.sub(r'([^!\?])(\?|!){2,}(\Z|[^!\?])', r'\1 \n\3', text)
    # Remove single question marks
    text = re.sub(r'([^!\?])\?(\Z|[^!\?])', r'\1 \n\2', text)
    # Remove single exclamation marks
    text = re.sub(r'([^!\?])!(\Z|[^!\?])', r'\1 \n\2', text)
    # Remove repeated (3+) letters: cooool --> cool, niiiiice --> niice
    text = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2', text)
    # Do it again in case we have coooooooollllllll --> cooll
    text = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2', text)
    # Remove smileys (big ones, small ones, happy or sad)
    text = re.sub(r' [8x;:=]-?(?:\)|\}|\]|>){2,}', r' ', text)
    text = re.sub(r' (?:[;:=]-?[\)\}\]d>])|(?:<3)', r' ', text)
    text = re.sub(r' [x:=]-?(?:\(|\[|\||\\|/|\{|<){2,}', r' ', text)
    text = re.sub(r' [x:=]-?[\(\[\|\\/\{<]', r' ', text)
    # Remove dots in words
    text = re.sub(r'(\w+)\.(\w+)', r'\1\2', text)

    clean_text = text

    # Split in phrases
    phrases = re.split(r'[;:\.()\n]', text)
    phrases = [re.findall(r'[\w%\*&#]+', ph) for ph in phrases]
    phrases = [ph for ph in phrases if ph]

    words = []
    for ph in phrases:
        words.extend(ph)

    # search for sequences of single letter words
    # like this ['f', 'u', 'c', 'k'] -> ['fuck']
    tmp = words
    words = []
    new_word = ''
    for word in tmp:
        if len(word) == 1:
            new_word = new_word + word
        else:
            if new_word:
                words.append(new_word)
                new_word = ''
            words.append(word)

    return {'original': original_text,
            'words': words,
            'ratio': ratio,
            'clean': clean_text,
            'class': label}


def partition_data(tokenized_tweets, partition):
    count = 0
    test_tweets = []
    train_tweets = []
    test_labels = []
    train_labels = []
    for tweet in tokenized_tweets:
        if count <= len(tokenized_tweets) / partition:
            test_tweets.append(tweet['clean'])
            test_labels.append(tweet['class'])
        else:
            train_tweets.append(tweet['clean'])
            train_labels.append(tweet['class'])
        count += 1
    return train_tweets, train_labels, test_tweets, test_labels


# def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
# plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(iris.target_names))
#     plt.xticks(tick_marks, iris.target_names, rotation=45)
#     plt.yticks(tick_marks, iris.target_names)
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


def get_confusion_matrix(expected, predicted):
    cm = confusion_matrix(expected, predicted)
    np.set_printoptions(precision=2)
    # print('Confusion matrix, without normalization')
    # print(cm)
    return cm
    # plt.figure()
    # plot_confusion_matrix(cm)


def get_f1_measure(expected, predicted):
    return f1_score(expected, predicted, average='macro')


def get_average_precision(expected, predicted):
    return (expected, predicted)


def get_average_recall(expected, predicted):
    return (expected, predicted)


def get_measures_for_each_class(expected, predicted):
    return precision_recall_fscore_support(expected, predicted, average='macro')


    # # Normalize the confusion matrix by row (i.e by the number of samples
    # # in each class)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print('Normalized confusion matrix')
    # print(cm_normalized)
    # plt.figure()
    # plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    #
    # plt.show()


def crossValidation(tweets, labels, partition):
    count = 0
    test_tweets = []
    train_tweets = []
    test_labels = []
    train_labels = []


    train_tweets, test_tweets, train_labels, test_labels = cv.train_test_split(tweets, labels, test_size=1/float(partition), random_state=0)
    test_tweets, validation_tweets, test_labels, validation_labels = cv.train_test_split(test_tweets, test_labels, test_size=1/float(2), random_state=0)

    return train_tweets, test_tweets, validation_tweets, train_labels, test_labels, validation_labels
