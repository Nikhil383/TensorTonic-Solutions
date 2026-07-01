def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    dim = len(points[0])

    # Sum vectors for each cluster
    sums = [[0] * dim for _ in range(k)]

    # Number of points in each cluster
    counts = [0] * k

    # Accumulate sums and counts
    for point, cluster in zip(points, assignments):
        counts[cluster] += 1
        for d in range(dim):
            sums[cluster][d] += point[d]

    # Compute centroids
    centroids = []
    for cluster in range(k):
        if counts[cluster] == 0:
            centroids.append([0.0] * dim)
        else:
            centroid = [
                sums[cluster][d] / counts[cluster]
                for d in range(dim)
            ]
            centroids.append(centroid)

    return centroids