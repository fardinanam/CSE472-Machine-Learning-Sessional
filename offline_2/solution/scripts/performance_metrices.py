from sklearn.metrics import confusion_matrix

def specificity_score(x, y):
    """
    Calculate the specificity score of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return tn / (tn + fp)

def accuracy_score(x, y):
    """
    Calculate the accuracy score of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return (tp + tn) / (tp + tn + fp + fn)

def precision_score(x, y):
    """
    Calculate the precision score of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return tp / (tp + fp)

def recall_score(x, y):
    """
    Calculate the recall score of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return tp / (tp + fn)

def f1_score(x, y):
    """
    Calculate the f1 score of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return (2 * tp) / (2 * tp + fp + fn)

def false_discovery_rate(x, y):
    """
    Calculate the false discovery rate of the confusion matrix
    """
    cm = confusion_matrix(x, y)
    tn, fp, fn, tp = cm.ravel()

    return fp / (tp + fp)