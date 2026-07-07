import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)

def evaluate(y_true, y_pred, y_score=None):
    """
    y_true, y_pred : binary 0/1 arrays (ground truth, hard prediction)
    y_score        : continuous anomaly score, needed for AUCROC/AUCPR
                      (if None, y_pred is reused, which degrades those two metrics)
    """
    results = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }
    score = y_score if y_score is not None else y_pred
    results["aucroc"] = roc_auc_score(y_true, score)
    results["aucpr"]  = average_precision_score(y_true, score)
    return results


# quick smoke test
if __name__ == "__main__":
    y_true = np.array([0,0,0,1,1,0,1,0])
    y_pred = np.array([0,0,1,1,0,0,1,0])
    print(evaluate(y_true, y_pred))