import numpy as np

def average_precision_at_k(preds, labels, k=5):
    score = 0.0
    hits = 0

    for i in range(min(k, len(preds))):
        if labels.get((preds[i]), 0) == 1:
            hits += 1
            score += hits / (i + 1)

    return score / min(k, len(preds))


def mean_average_precision(all_preds, all_labels, k=5):
    scores = []
    for qid in all_preds:
        preds = all_preds[qid]
        labels = all_labels[qid]
        scores.append(average_precision_at_k(preds, labels, k))
    return np.mean(scores)