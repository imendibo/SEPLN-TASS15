import numpy as np
import os
if __name__ == '__main__':
    print os.getcwd()
    train_file = 'train_results.csv'
    train_set = np.genfromtxt(train_file, delimiter=',')

    train_label_file = 'train_labels.csv'
    train_label = np.genfromtxt(train_label_file, delimiter=',')

    test_file = 'test_results.csv'
    train_set = np.genfromtxt(test_file, delimiter=',')

    test_label_file = 'test_labels.csv'
    train_label = np.genfromtxt(test_label_file, delimiter=',')

