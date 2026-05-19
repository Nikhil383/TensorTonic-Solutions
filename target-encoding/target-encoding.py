def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Store sum and count for each category
    category_stats = {}

    for cat, target in zip(categories, targets):
        if cat not in category_stats:
            category_stats[cat] = [0, 0]  # [sum, count]

        category_stats[cat][0] += target
        category_stats[cat][1] += 1

    # Compute mean target for each category
    category_mean = {
        cat: total / count
        for cat, (total, count) in category_stats.items()
    }

    # Replace categories with their mean target value
    encoded = [category_mean[cat] for cat in categories]

    return encoded