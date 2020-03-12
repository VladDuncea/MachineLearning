import math
from sklearn import preprocessing
from sklearn import svm
import numpy as np

def calc_accuracy(predicted, true):
    accuracy = (predicted == true).mean()
    return accuracy

# -----------------------------
# punctul 2
def normalize_data(train_data, test_data, type=None):
    if type == 'standard':
        stand = preprocessing.StandardScaler()
        stand.fit(train_data)
        train_data = stand.transform(train_data)
        stand.fit(test_data)
        test_data = stand.transform(test_data)
    elif type == 'min_max':
        x = 1+1
    # posibil sa trebuiasca sa fac l1,l2 cu numpy !!!
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
            if norm_test != 0:
                test_data[i] /= norm_test

    return train_data, test_data


class BagOfWords:

    def __init__(self):
        self.dictData = dict()
        self.word_list = []

    def build_vocabulary(self, data):
        index =0
        for sentence in data:
            for word in sentence:
                if word not in self.dictData:
                    self.dictData[word] = index
                    self.word_list.append(word)
                    index += 1
        return self.dictData

    def get_features(self, data):
        features = np.zeros((len(data), len(self.dictData)))
        for i in range(len(data)):
            for word in data[i]:
                if word in self.dictData:
                    features[i, self.dictData[word]] += 1
        return features

# load data
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
train_sentences = np.load('data/training_sentences.npy')
train_labels = np.load('data/training_labels.npy')
test_sentences = np.load('data/test_sentences.npy')
test_labels = np.load('data/test_labels.npy')
np.load = np_load_old


# create class
bagofwords = BagOfWords()
# build train dict
dict_data = bagofwords.build_vocabulary(train_sentences)
print("Lungime dictionar:" + str(len(dict_data)))

# get features
features_train = bagofwords.get_features(train_sentences)
features_test = bagofwords.get_features(test_sentences)

normalized_train, normalized_test = normalize_data(features_train, features_test, "l2")
print(normalized_train)
print(normalized_test)

# SVM model
C_param = 1
svm_model = svm.SVC(C_param, "linear") # kernel liniar
svm_model.fit(normalized_train, train_labels) # train
predicted_labels = svm_model.predict(normalized_test) # predict

print("Accuracy: " + str(calc_accuracy(predicted_labels, test_labels)))

