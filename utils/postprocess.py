from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import math


def get_accuracy(predict, target, type):
    conf_matrix = confusion_matrix(target, predict)
    cls_acc = []
    for i in range(conf_matrix.shape[0]):
        if np.sum(conf_matrix[i]) == 0:
            cls_acc.append(0.0)
        else:
            cls_acc.append(conf_matrix[i][i] / np.sum(conf_matrix[i]))
    total_acc = accuracy_score(target, predict)
    f1 = f1_score(target, predict, average=type)
    precision = precision_score(target, predict, average=type)
    recall = recall_score(target, predict, average=type)
    cc = np.corrcoef(predict, target)
    MSE = np.square(np.subtract(target, predict)).mean()
    RMSE = math.sqrt(MSE)


    return {'total_accuracy': total_acc, 'f1_score': f1,
            'precision_score': precision, 'recall_score': recall,
            'class_accuracies': cls_acc, 'correlation coefficient': cc,
            'RMSE': RMSE}

