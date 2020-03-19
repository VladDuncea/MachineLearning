import sklearn
from sklearn import preprocessing
from sklearn import svm
import numpy as np
from sklearn.metrics import f1_score


# -----------------------------
# punctul 2
def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()
    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()
    elif type == 'l1':
        scaler = preprocessing.Normalizer('l1')
    elif type == 'l2':
        scaler = preprocessing.Normalizer('l2')

    if scaler is not None:
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

    return train_data, test_data


class BagOfWords:

    def __init__(self):
        self.dictData = dict()
        self.word_list = []
        self.dict_length = 0

    def build_vocabulary(self, data):
        for sentence in data:
            for word in sentence:
                if word not in self.dictData:
                    self.dictData[word] = self.dict_length
                    self.word_list.append(word)
                    self.dict_length += 1
        self.word_list = np.array(self.word_list)

    def get_features(self, data):
        features = np.zeros((len(data), len(self.dictData)))
        for i in range(len(data)):
            for word in data[i]:
                if word in self.dictData:
                    features[i, self.dictData[word]] += 1
        return features


# load data
train_sentences = np.load('data/training_sentences.npy', allow_pickle=True)
train_labels = np.load('data/training_labels.npy', allow_pickle=True)
test_sentences = np.load('data/test_sentences.npy', allow_pickle=True)
test_labels = np.load('data/test_labels.npy', allow_pickle=True)

# create class
bagofwords = BagOfWords()
# build train dict
bagofwords.build_vocabulary(train_sentences)
print("Lungime dictionar:" + str(bagofwords.dict_length))

# get features
features_train = bagofwords.get_features(train_sentences)
features_test = bagofwords.get_features(test_sentences)

normalized_train, normalized_test = normalize_data(features_train, features_test, "l2")
# print(normalized_train)
# print(normalized_test)

# SVM model
C_param = 1
svm_model = svm.LinearSVC(C=C_param)  # kernel liniar
svm_model.fit(normalized_train, train_labels)  # train
predicted_labels = svm_model.predict(normalized_test)  # predict

print("Accuracy: " + str(sklearn.metrics.accuracy_score(predicted_labels, test_labels)))
print("F1: " + str(f1_score(predicted_labels, test_labels)))

words = np.array(bagofwords.word_list)
weights = np.squeeze(svm_model.coef_)
indexes = np.argsort(weights)
print("the first 10 negative (spam) words are", words[indexes[-10:]])
print("the first 10 positive (not spam) words are", words[indexes[:10]])
