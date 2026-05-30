import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)

    silhouettes = []

    for i in range(n_samples):
        current_label = labels[i]

        # Points in the same cluster (excluding itself)
        same_cluster = np.where(labels == current_label)[0]
        same_cluster = same_cluster[same_cluster != i]

        # a(i): mean intra-cluster distance
        if len(same_cluster) == 0:
            silhouettes.append(0.0)
            continue

        a_i = np.mean(
            np.linalg.norm(X[i] - X[same_cluster], axis=1)
        )

        # b(i): minimum mean distance to another cluster
        b_i = np.inf

        for label in unique_labels:
            if label == current_label:
                continue

            other_cluster = np.where(labels == label)[0]

            avg_dist = np.mean(
                np.linalg.norm(X[i] - X[other_cluster], axis=1)
            )

            b_i = min(b_i, avg_dist)

        s_i = (b_i - a_i) / max(a_i, b_i)
        silhouettes.append(s_i)

    return float(np.mean(silhouettes))