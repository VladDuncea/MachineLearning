from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.utils import shuffle


def getdata_folds(data,labels,nr_folds,id_test):
    folds = np.array_split(data, nr_folds)
    label_folds = np.array_split(labels, nr_folds)
    training_data = []
    training_labels = []
    for j in range(nr_folds):
        if j != id_test:
            if len(training_data) == 0:
                training_data = folds[j]
                training_labels = label_folds[j]
            else:
                training_data = np.concatenate((training_data, folds[j]))
                training_labels = np.concatenate((training_labels, label_folds[j]))
    test_data = folds[id_test]
    test_labels = label_folds[id_test]
    return training_data,training_labels,test_data,test_labels

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


# load training data
data = np.load('data/training_data.npy')
prices = np.load('data/prices.npy')

# shuffle
data, prices = shuffle(data, prices, random_state=0)

# PUNCTUL 2
print("------------LINEAR-------------")
nr_folds = 3

linear_regression_model = LinearRegression()
mse_values = np.zeros(nr_folds)
mae_values = np.zeros(nr_folds)
for i in range(nr_folds):
    training_data,training_labels,test_data,test_labels = getdata_folds(data,prices,nr_folds,i)
    training_data,test_data = normalize_data(training_data,test_data,"min_max")

    linear_regression_model.fit(training_data,training_labels)
    predictions = linear_regression_model.predict(test_data)

    # calcularea valorii MSE și MAE
    mse_values[i] = mean_squared_error(test_labels, predictions)
    mae_values[i] = mean_absolute_error(test_labels, predictions)

print("MSE: " + str(mse_values.mean()))
print("MAE: " + str(mae_values.mean()))

# PUNCTUL 3
print("------------RIDGE-------------")

mse_values = np.zeros(nr_folds)
mae_values = np.zeros(nr_folds)
val_alpha = [1,10,100,1000]
for alpha in val_alpha:
    ridge_regression_model = Ridge(alpha=alpha)
    for i in range(nr_folds):
        training_data, training_labels, test_data, test_labels = getdata_folds(data,prices,nr_folds,i)
        training_data,test_data = normalize_data(training_data,test_data,"min_max")

        ridge_regression_model.fit(training_data,training_labels)
        predictions = ridge_regression_model.predict(test_data)

        # calcularea valorii MSE și MAE
        mse_values[i] = mean_squared_error(test_labels, predictions)
        mae_values[i] = mean_absolute_error(test_labels, predictions)
    print("ALPHA: " + str(alpha))
    print("MSE: " + str(mse_values.mean()))
    print("MAE: " + str(mae_values.mean()))

# PUNCTUL 4
print("------------PUNCTUL 4 -------------")

ridge_regression_model = Ridge(alpha=1)
training_data,nop = normalize_data(data,data,"min_max")
ridge_regression_model.fit(training_data, prices)
print("Coef:")
print(ridge_regression_model.coef_)
indexes = np.argsort(ridge_regression_model.coef_)
descr = ["An fabr","nr kilometrii","consum","motor","putere","nr_locuri","nr_proprietari","combustibil","combustibil","combustibil","combustibil","combustibil","transmisie","transmisie"]
print("Cea mai semnificativa coloana :" + str(descr[indexes[-1]]))
print("A doua cea mai semnificativa coloana :" + str(descr[indexes[-2]]))
print("Cea mai nesemnificativa coloana :" + str(descr[indexes[0]]))


