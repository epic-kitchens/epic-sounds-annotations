#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch
import numpy as np
from scipy import stats
import sklearn.metrics as met
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct

def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]

def map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes, or num_examples.
    Returns:
        mean_ap (int): final mAP score.
    """
    # Convert labels to one hot vector
    labels = np.eye(preds.shape[1])[labels] if labels.ndim == 1 else labels
    
    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = met.average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap

def per_class_accuracy(preds, labels):
    preds_inds = preds.argmax(axis=1)
    matrix = met.confusion_matrix(labels, preds_inds)
    class_counts = matrix.sum(axis=1)
    correct = matrix.diagonal()[class_counts != 0]
    class_counts = class_counts[class_counts != 0]
    return correct / class_counts

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including per class accuracy, mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Class-wise statistics
    for k in range(classes_num):
        # Average precision
        avg_precision = met.average_precision_score(
                target[:, k], output[:, k], average=None
            )

        # AUC
        auc = met.roc_auc_score(
                target[:, k], output[:, k], average=None
            )

        # Precisions, recalls
        (precisions, recalls, thresholds) = met.precision_recall_curve(
                target[:, k], output[:, k]
            )

        # FPR, TPR
        (fpr, tpr, thresholds) = met.roc_curve(
                target[:, k], output[:, k]
            )

        save_every_steps = 1000   # Sample statistics to reduce size
        stats_dict = {
                'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc
            }
        stats.append(stats_dict)

    return stats

def get_stats(preds, labels):
    per_class_acc = per_class_accuracy(preds, labels)

    # Convert labels to one hot vector
    labels = np.eye(preds.shape[1])[labels] if labels.ndim == 1 else labels

    # Only calculate for seen classes
    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]

    stats = calculate_stats(preds, labels)
    # Write out to log
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    m_PCA = np.mean([class_acc for class_acc in per_class_acc])
    per_class_acc = np.around(per_class_acc, 5)

    stats_dict = {}
    stats_dict["mAP"] = mAP
    stats_dict["mAUC"] = mAUC
    stats_dict["d_prime"] = d_prime(mAUC)
    stats_dict["PCA"] = per_class_acc.tolist()
    stats_dict["mPCA"] = m_PCA
    stats_dict["seen_classes"] = labels.shape[1]

    return stats_dict