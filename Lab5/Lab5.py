import math
from sklearn import preprocessing
import numpy as np


def normalize_data(train_data, test_data, type=None):
    if type == 'standard':
        stand = preprocessing.StandardScaler()
        stand.fit(train_data)
        train_data = stand.transform(train_data)
        stand.fit(test_data)
        test_data = stand.transform(test_data)
    elif type == 'min_max':
        x = 1+1
    elif type == 'l1':
        for i in range(len(train_data)):
            norm_train = 0
            norm_test = 0
            for j in range(len(train_data[i])):
                norm_train += abs(train_data[i, j])
                norm_test += abs(train_data[i, j])
            train_data[i] /= norm_train
            test_data[i] /= norm_test
    elif type == 'l2':
        for i in range(len(test_data)):
            norm_test = 0
            norm_train = 0
            for j in range(len(test_data[i])):
                norm_test += math.sqrt((test_data[i, j])**2)
                norm_train += math.sqrt((train_data[i, j])**2)
            train_data[i] /= norm_train
            test_data[i] /= norm_test

    return train_data, test_data


np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

train_sentences = np.load('data/training_sentences.npy')
train_labels = np.load('data/training_labels.npy')
test_sentences = np.load('data/test_sentences.npy')
test_labels = np.load('data/test_labels.npy')

np.load = np_load_old

#train_sentences, test_sentences = normalize_data(train_sentences,test_sentences,'l1')
print(train_sentences)
print(test_sentences)