import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.

    Parameters
    ----------
    y_true_list : list of array-like
        Binary relevance labels for each query.
    y_score_list : list of array-like
        Predicted scores for each query.
    k : int or None, default=None
        Compute AP@k. If None, use all ranked items.

    Returns
    -------
    tuple
        (map_value, ap_per_query)

        map_value : float
            Mean Average Precision across all queries.

        ap_per_query : list of float
            Average Precision for each query.
    """

    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        # Total relevant documents in the ground truth
        total_relevant = np.sum(y_true)

        # Sort by descending predicted score
        order = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[order]

        # Apply top-k if specified
        if k is not None:
            y_true_sorted = y_true_sorted[:k]

        # No relevant documents
        if total_relevant == 0:
            ap_per_query.append(0.0)
            continue

        hits = 0
        precision_sum = 0.0

        for rank, rel in enumerate(y_true_sorted, start=1):
            if rel:
                hits += 1
                precision_sum += hits / rank

        ap = precision_sum / total_relevant
        ap_per_query.append(ap)

    map_value = np.mean(ap_per_query) if ap_per_query else 0.0

    return map_value, ap_per_query