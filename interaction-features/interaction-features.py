def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    # Write code here
    result = []

    for row in X:
        interaction_list = []

        d = len(row)

        for i in range(d):
            for j in range(i + 1, d):
                interaction_list.append(row[i] * row[j])

        result.append(list(row) + interaction_list)

    return result