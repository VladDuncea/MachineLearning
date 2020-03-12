import numpy as np
import math
import matplotlib.pyplot as plt

def calc_accuracy(predicted, true):
    accuracy = (predicted == true).mean()
    return accuracy


class KnnClassifier:

    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        n = len(self.train_images)
        val = np.zeros(n)

        if metric == 'l1':
            val = np.sum(abs(self.train_images-test_image), axis=1)
        elif metric == 'l2':
            val = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))

        labels_index = val.argsort()[:num_neighbors]  # sort matrix by first column and preserve rows
        labels = self.train_labels[labels_index]    # returns most frequent label, if all are equal returns first one
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
calc_acc_l1 = True
calc_acc_l2 = True

accuracy_l1 = np.zeros(len(nearest_neigh))
accuracy_l2 = np.zeros(len(nearest_neigh))
# nr of images
nr = len(test_images)

if calc_acc_l1:
    j = 0
    for nn in nearest_neigh:
        for i in range(nr):
            predictions[i] = classifier.classify_image(test_images[i], nn, 'l1')
        accuracy_l1[j] = calc_accuracy(predictions, test_labels)
        print("L1:Accuracy on "+str(nn)+" nn: " + str(accuracy_l1[j]))
        # np.savetxt('predictii_3nn_l1_mnist.txt', predictions.astype(int))  # salveaza array-ul y in fisier
        j += 1

    # save accuracy
    np.savetxt('acuratete_l1.txt', accuracy_l1)
else:
    accuracy_l1 = np.loadtxt("acuratete_l1.txt")

if calc_acc_l2:
    j = 0
    for nn in nearest_neigh:
        for i in range(nr):
            predictions[i] = classifier.classify_image(test_images[i], nn, 'l2')

        accuracy_l2[j] = calc_accuracy(predictions, test_labels)
        print("L2:Accuracy on " + str(nn) + " nn: " + str(accuracy_l2[j]))
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


# import numpy as np
# import pdb
#
#
# class Knn_classifier:
#     def __init__(self, train_images, train_labels):
#         self.train_images = train_images
#         self.train_labels = train_labels
#
#     def classify_image(self, test_image, num_neighbors=3, metric='l2'):
#         if (metric == 'l2'):
#             distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))
#         elif (metric == 'l1'):
#             distances = np.sum(abs(self.train_images - test_image), axis=1)
#         else:
#             raise ('Unknown metric!')
#         sort_index = np.argsort(distances)
#         sort_index = sort_index[:num_neighbors]
#         nearest_labels = self.train_labels[sort_index]
#         hist = np.bincount(nearest_labels)
#         return np.argmax(hist)
#
#     def classify_images(self, test_images, num_neighbors=3, metric='l2'):
#         num_imgs = test_images.shape[0]
#         predicted_labels = np.zeros((num_imgs), np.int)
#         for i in range(num_imgs):
#             predicted_labels[i] = self.classify_image(test_images[i, :], num_neighbors=num_neighbors, metric=metric)
#         return predicted_labels
#
#
# def accuracy_score(y_true, y_pred):
#     return (y_pred == y_true).mean()
#
#
# train_images = np.loadtxt('train_images.txt')  # incarcam imaginile
# train_labels = np.loadtxt('train_labels.txt').astype(int)  # incarcam etichetele avand
# test_images = np.loadtxt('test_images.txt')  # incarcam imaginile
# test_labels = np.loadtxt('test_labels.txt').astype(int)  # incarcam etichetele avand
#
# classifier = Knn_classifier(train_images, train_labels)
# predicted_labels = classifier.classify_images(test_images, 3, metric='l2')
# print(accuracy_score(test_labels, predicted_labels))
# np.savetxt('predictii_3nn_l2_mnist.txt', predicted_labels)
