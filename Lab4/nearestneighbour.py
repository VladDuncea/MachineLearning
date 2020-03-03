import numpy as np
import math


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

        return 0
