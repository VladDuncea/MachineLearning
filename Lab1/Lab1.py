y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


def accuracy_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return 0

    # for (t,p) in zip(y_true,y_pred):
    #    if t == p:
    #        sum = sum+1
    sum = [1 for (t, p) in zip(y_true, y_pred) if t == p]
    accuracy = float(len(sum)) / len(y_true)

    return accuracy


def precision_recall_score(y_true, y_pred):
    tp = 0.0
    fp = 0
    fn = 0
    for (t, p) in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp = tp + 1
        elif t == 0 and p == 1:
            fp = fp + 1
        elif t == 1 and p == 0:
            fn = fn + 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def mse(y_true, y_pred):
    sum = 0.0
    for (t, p) in zip(y_true, y_pred):
        sum = sum + (p - t) ** 2
    sum = sum / len(y_true)
    return sum


def mae(y_true, y_pred):
    sum = 0.0
    for (t, p) in zip(y_true, y_pred):
        sum = sum + abs(t - p)
    sum = sum / len(y_true)
    return sum


print(accuracy_score(y_true, y_pred))
print(precision_recall_score(y_true, y_pred))
print(mse(y_true, y_pred))
print(mae(y_true, y_pred))