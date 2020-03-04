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


def calc_accuracy(predicted, true):
    accuracy = 0
    for (pred, tru) in zip(predicted, true):
        if pred == tru:
            accuracy += 1
    return round((accuracy / len(predictions))*100, 2)


class KnnClassifier:

    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_images(self, test_image, num_neighbors=3, metric='l2'):
        n = len(self.train_images)
        val = np.zeros((n, 2))
        val[:, 1] = self.train_labels

        if metric == 'l1':
            for i in range(n):
                val[i, 0] = dist_1(self.train_images[i], test_image)
        elif metric == 'l2':
            for i in range(n):
                val[i, 0] = dist_2(self.train_images[i], test_image)

        labels = val[val[:, 0].argsort()[0:num_neighbors], 1].astype(int)  # sort matrix by first column and preserve rows
        # returns most frequent label, if all are equal returns first one
        prediction = np.bincount(labels).argmax()
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

#build or read accuracy ?
calc_acc_l1 = False
calc_acc_l2 = False

accuracy_l1 = np.zeros(len(nearest_neigh))
accuracy_l2 = np.zeros(len(nearest_neigh))

if calc_acc_l1:
    j = 0
    for nn in nearest_neigh:
        for i in range(len(test_images)):
            predictions[i] = classifier.classify_images(test_images[i], nn, 'l1')
            print("L1-NN:" + str(nn) + "-" + str(i))

        accuracy_l1[j] = calc_accuracy(predictions, test_labels)

        print("Accuracy on"+str(nn)+" nn: " + str(accuracy_l1[j]))
        # np.savetxt('predictii_3nn_l1_mnist.txt', predictions.astype(int))  # salveaza array-ul y in fisier
        j += 1

    # save accuracy
    np.savetxt('acuratete_l1.txt', accuracy_l1)
else:
    accuracy_l1 = np.loadtxt("acuratete_l1.txt")

if calc_acc_l2:
    j = 0
    for nn in nearest_neigh:
        for i in range(len(test_images)):
            predictions[i] = classifier.classify_images(test_images[i], nn, 'l2')
            print("L2-NN:" + str(nn) + "-" + str(i))

        accuracy_l2[j] = calc_accuracy(predictions, test_labels)

        print("Accuracy on" + str(nn) + " nn: " + str(accuracy_l2[j]))
        # np.savetxt('predictii_3nn_l2_mnist.txt', predictions.astype(int))  # salveaza array-ul y in fisier
        j += 1

    # save accuracy
    np.savetxt('acuratete_l2.txt', accuracy_l2)
else:
    accuracy_l2 = np.loadtxt("acuratete_l2.txt")

# plot accuracy
plt.plot(nearest_neigh, accuracy_l1)
plt.plot(nearest_neigh, accuracy_l2)
plt.title('Acuratete')
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy')
plt.legend(['L1', 'L2'])
plt.show()



