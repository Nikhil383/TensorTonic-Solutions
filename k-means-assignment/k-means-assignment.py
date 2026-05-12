def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    assignments = []

    # Loop through each point
    for p in points:

        best_dist = float('inf')
        best_idx = 0

        # Check distance to every centroid
        for i, c in enumerate(centroids):

            D = len(p)

            # Squared Euclidean distance
            squared_distance = sum(
                (p[d] - c[d]) ** 2
                for d in range(D)
            )

            # Keep nearest centroid
            if squared_distance < best_dist:
                best_dist = squared_distance
                best_idx = i

        assignments.append(best_idx)

    return assignments
    