import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

    cm = np.zeros((num_classes, num_classes), dtype=float)

    # Fill confusion matrix
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Normalization
    if normalize == 'none':
        return cm.astype(int)

    elif normalize == 'true':
        row_sum = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    elif normalize == 'pred':
        col_sum = cm.sum(axis=0, keepdims=True)
        return np.divide(cm, col_sum, out=np.zeros_like(cm), where=col_sum != 0)

    elif normalize == 'all':
        total = cm.sum()
        return cm / total if total != 0 else cm

    else:
        raise ValueError("normalize must be one of {'none', 'true', 'pred', 'all'}")