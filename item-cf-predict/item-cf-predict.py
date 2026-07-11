def item_cf_predict(user_ratings, item_similarities, target):
    """
    Predict the rating using item-based collaborative filtering.
    """
    # Write code here
    numerator = 0.0
    denominator = 0.0

    for i in range(len(user_ratings)):
        # Skip the target item
        if i == target:
            continue

        # Skip unrated items
        if user_ratings[i] == 0:
            continue

        # Skip non-positive similarities
        if item_similarities[i] <= 0:
            continue

        numerator += item_similarities[i] * user_ratings[i]
        denominator += item_similarities[i]

    if denominator == 0:
        return 0.0

    return numerator / denominator