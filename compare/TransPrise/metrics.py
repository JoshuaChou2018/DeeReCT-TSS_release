"""
Metrics for training and prediction
"""
from sklearn.metrics import roc_curve, auc
import numpy as np


def sensetivity(tp, fn):
    """
    
    :param tp: true positives results
    :param fn: false negative results
    :return: sensetivity
    """
    return tp/(tp + fn)


def specificity(tn, fp):
    """
    
    :param tn: true negative results
    :param fp: false positive results
    :return: specificity
    """

    return tn/(tn+fp)


def accuracy(tp, tn, fp, fn):
    """
    
    :param tp: true positives results
    :param tn: true negatives results
    :param fp: false positives results
    :param fn: false negatives results
    :return: accuracy
    """

    return (tp+tn)/(tp+tn+fp+fn)


def results(predictions, true):
    """
    
    :param predictions: predicted values
    :param true: real values
    :return: true positives, true negatives, false positives, false negatives
    """
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(predictions)):
        result = 1 if predictions[i] >= 0.5 else 0
        if result == true[i]:
            if result == 1:
                tp += 1
            else:
                tn += 1
        else:
            if result == 1:
                fp += 1
            else:
                fn += 1

    return tp, tn, fp, fn


def cc(tp, tn, fp, fn):
    """
    
    :param tp: true positives results
    :param tn: true negatives results
    :param fp: false positives results
    :param fn: false negatives results
    :return: correlation coefficient
    """

    return (tp*tn - fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5


def roc_auc(predictions, true):
    """
    
    :param predictions: predicted values
    :param true: true values
    :return: false positive rate, true positive rate, roc_auc
    """

    fpr, tpr, thresholds = roc_curve(true, predictions)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score


def mse(predictions, test_answers):
    """
    
    :param predictions: predicted values
    :param test_answers: real values
    :return: mean squared error
    """

    return np.mean((predictions.squeeze() - test_answers) ** 2)


def rmse(predictions, true):
    """
    
    :param predictions: predicted values
    :param true: real values
    :return: rooted mean squared error or mean absolute error
    """

    return np.mean(((predictions.squeeze() - true) ** 2) ** 0.5)


def all_class_metrics(predictions, true):
    """
    
    :param predictions: predicted values
    :param true: true values
    :return: print all metrics
    """

    tp, tn, fp, fn = results(predictions, true)
    print('TP:', tp, '; TN:', tn, '; FP:', fp, '; FN:', fn)
    print('Accuracy:', accuracy(tp, tn, fp, fn))
    print('Sensetivity:', sensetivity(tp, fn))
    print('Specificity:', specificity(tn, fp))
    print('Correlation coefficient:', cc(tp, tn, fp, fn))
    print('AUC:', roc_auc(predictions, true)[2])
