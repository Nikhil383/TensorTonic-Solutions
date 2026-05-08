def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    # Write code here
    result = []

    for req in requests:
        user_id = req["user_id"]

        # Get offline features or defaults
        offline_features = feature_store.get(user_id, defaults)

        # Get online features
        online_features = req.get("online_features", {})

        # Merge both feature sets
        merged = {**offline_features, **online_features}

        result.append(merged)

    return result