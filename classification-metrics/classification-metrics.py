import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    else:
        classes = np.unique(np.concatenate([y_true, y_pred]))

        precisions = []
        recalls = []
        f1s = []
        supports = []

        total_tp = total_fp = total_fn = 0

        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            support = np.sum(y_true == cls)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        supports = np.array(supports)

        if average == "micro":
            precision = (
                total_tp / (total_tp + total_fp)
                if (total_tp + total_fp) > 0
                else 0.0
            )
            recall = (
                total_tp / (total_tp + total_fn)
                if (total_tp + total_fn) > 0
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        elif average == "macro":
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)

        elif average == "weighted":
            weights = supports / supports.sum()
            precision = np.sum(weights * np.array(precisions))
            recall = np.sum(weights * np.array(recalls))
            f1 = np.sum(weights * np.array(f1s))

        else:
            raise ValueError(
                "average must be one of: 'binary', 'micro', 'macro', 'weighted'"
            )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    