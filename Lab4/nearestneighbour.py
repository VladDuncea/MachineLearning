import numpy as np
import math
import matplotlib.pyplot as plt


def dist_1(X, Y):
    dist = 0
    for (x, y) in zip(X, Y):
        dist += abs(x - y)
    return dist


def dist_2(X, Y):
    dist = 0
    for (x, y) in zip(X, Y):
        dist += (x - y) * (x - y)
    dist = math.sqrt(dist)
    return dist


class KnnClassifier:

    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_images(self, test_image, num_neighbors=3, metric='l2'):
        prediction = 0
        val = np.zeros((len(self.train_labels), 2))
        val[:, 1] = self.train_labels

        if metric == 'l1':
            for i in range(len(self.train_images)):
                val[i, 0] = dist_1(self.train_images[i], test_image)
        elif metric == 'l2':
            for i in range(len(self.train_images)):
                val[i, 0] = dist_2(self.train_images[i], test_image)

        val = val[val[:, 0].argsort()]  # sort matrix by first column and preserve rows
        # returns most frequent label, if all are equal returns first one
        prediction = np.bincount(val[0:num_neighbors, 1].astype(int)).argmax()
        return prediction


# ------------------------------------
# punctul 3 + 4

train_images = np.loadtxt('../Lab3/data/train_images.txt')
train_labels = np.loadtxt('../Lab3/data/train_labels.txt', 'int')
test_images = np.loadtxt('../Lab3/data/test_images.txt')
test_labels = np.loadtxt('../Lab3/data/test_labels.txt', 'int')

classifier = KnnClassifier(train_images, train_labels)

predictions = np.zeros(len(test_images))

nearest_neigh = [1, 3, 5, 7, 9]
accuracy = np.zeros(len(nearest_neigh))
j = 0
for nn in nearest_neigh:
    for i in range(len(test_images)):
        predictions[i] = classifier.classify_images(test_images[i], nn, 'l2')
        print("NN:" + str(nn) + "-" + str(i))

    for (pred, tru) in zip(predictions, test_labels):
        if pred == tru:
            accuracy[j] += 1
    accuracy[j] = accuracy[j] / len(predictions)

    print("Accuracy on"+str(nn)+" nn: " + str(accuracy))
    # np.savetxt('predictii_3nn_l2_mnist.txt', predictions)  # salveaza array-ul y in fisier
    j += 1
# plot accuracy
plt.plot(nearest_neigh, accuracy)
plt.title('Acuratete')
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy')
plt.show()

np.savetxt('acuratete_l2.txt', accuracy)