import numpy as np
import os
import json
import random
import sklearn.metrics

def micro_f1(truth, pred):
    return sklearn.metrics.f1_score(truth,pred,average="micro")

def macro_precision(truth, pred):
    return sklearn.metrics.precision_score(truth,pred,average="macro")

def macro_recall(truth, pred):
    return sklearn.metrics.recall_score(truth,pred,average="macro")

def macro_f1(truth, pred):
    return sklearn.metrics.f1_score(truth,pred,average="macro")

def accuracy(truth, pred):
    return sklearn.metrics.accuracy_score(truth,pred)

def RMSE(truth, pred):
    return np.sqrt(np.mean((np.array(pred)-np.array(truth))**2))

def MSE(truth, pred):
    return np.mean((np.array(truth)-np.array(pred))**2)

accuracy_list = {
    "macro-f1": macro_f1,
    "macro-precision": macro_precision,
    "macro-recall": macro_recall,
    "micro-f1": micro_f1,
    "accuracy": accuracy,
    "MSE": MSE,
    "RMSE": RMSE,
}


def get_accuracy(accuracy_name):
    if accuracy_name in accuracy_list.keys():
        return accuracy_list[accuracy_name]
    else:
        raise NotImplementedError
