import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt', 'int')
test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', 'int')

# -----------------------------------------------------
# punctul 2
num_bins = 5
bins = np.linspace(start=0, stop=255, num=num_bins)  # returneaza intervalele


def values_to_bins(matrix, _bins):
    discreet_matrix = np.digitize(matrix, _bins) - 1
    return discreet_matrix


discreet_train = values_to_bins(train_images, bins)
discreet_test = values_to_bins(test_images, bins)

# -----------------------------------------------------
# punctul 3

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(discreet_train, train_labels)
naive_bayes_model.predict(discreet_test)

score = naive_bayes_model.score(discreet_test, test_labels)
print("Scorul: " + str(score))

# -----------------------------------------------------
# punctul 4
num_bins_vect = np.array([3, 5, 7, 9, 11])

for num_bins in num_bins_vect:
    # calculeaza intervalele
    bins = np.linspace(start=0, stop=255, num=num_bins)

    # discretizeaza valorile
    discreet_train = values_to_bins(train_images, bins)
    discreet_test = values_to_bins(test_images, bins)

    # construieste si testeaza clasificatorul
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(discreet_train, train_labels)
    naive_bayes_model.predict(discreet_test)

    score = naive_bayes_model.score(discreet_test, test_labels)
    print("Scorul pentru " + str(num_bins) + " intervale: " + str(score))

# -----------------------------------------------------
# punctul 5

num_bins = 11

# calculeaza intervalele
bins = np.linspace(start=0, stop=255, num=num_bins)

# discretizeaza valorile
discreet_train = values_to_bins(train_images, bins)
discreet_test = values_to_bins(test_images, bins)

# construieste si testeaza clasificatorul
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(discreet_train, train_labels)

predictions = naive_bayes_model.predict(discreet_test)

cnt = 0
for i in range(len(predictions)):
    if predictions[i] != test_labels[i]:
        image = test_images[i, :]
        image = np.reshape(image, (28, 28))
        plt.imshow(image.astype(np.uint8), cmap='gray')
        plt.title("Aceasta imagine a fost clasificata ca: " + str(predictions[i]) + " dar era: " + str(test_labels[i]))
        plt.show()
        cnt += 1
    if cnt >= 10:
        break


# -----------------------------------------------------
# punctul 6

def confusion_matrix(y_true, y_pred):
    cmatrix = np.zeros((10, 10))
    for (t, p) in zip(y_true, y_pred):
        cmatrix[t, p] += 1
    return cmatrix

print(confusion_matrix(test_labels, predictions))
