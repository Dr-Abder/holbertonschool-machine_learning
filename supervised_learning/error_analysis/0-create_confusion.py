#!/usr/bin/env python3

import numpy as np

def create_confusion_matrix(labels, logits):

    classes = labels.shape[1]

    confusion_matrix = np.zeros((classes, classes))

    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    for i in range(len(true_labels)):

        true_classe = true_labels[i]
        pred_classe = pred_labels[i]

        confusion_matrix[true_classe, pred_classe] += 1

    return confusion_matrix