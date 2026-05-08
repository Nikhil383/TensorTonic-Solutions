def promote_model(models):
    """
    Decide which model version to promote to production.

    Priority:
    1. Higher accuracy
    2. Lower latency
    3. More recent timestamp
    """

    best_model = sorted(
        models,
        key=lambda m: (
            -m["accuracy"],
            m["latency"],
            -int(m["timestamp"].replace("-", ""))
        )
    )[0]

    return best_model["name"]