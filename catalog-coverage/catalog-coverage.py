def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    if n_items == 0:
        return 0.0
    
    unique_items = set()
    
    # Collect unique recommended items
    for rec_list in recommendations:
        unique_items.update(rec_list)
    
    # Coverage = unique recommended items / total catalog items
    return len(unique_items) / n_items