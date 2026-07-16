import math

def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.

    Parameters:
        recommendations (list): List of recommended item indices.
        item_counts (list or dict): Number of users who interacted with each item.
        n_users (int): Total number of users.

    Returns:
        float: Average novelty score.
    """
    if not recommendations:
        return 0.0

    novelty = 0

    for item in recommendations:
        novelty += -math.log2(item_counts[item] / n_users)

    return novelty / len(recommendations)