def robust_scaling(values):
    if len(values) == 1:
        return [0.0]

    original = values[:]
    arr = sorted(values)

    def median(lst):
        n = len(lst)
        if n == 0:
            return 0.0
        if n % 2 == 0:
            return (lst[n//2 - 1] + lst[n//2]) / 2.0
        return float(lst[n//2])

    n = len(arr)

    # Overall median
    med = median(arr)

    # Lower and upper halves
    if n % 2 == 0:
        lower = arr[:n//2]
        upper = arr[n//2:]
    else:
        lower = arr[:n//2]
        upper = arr[n//2 + 1:]

    q1 = median(lower)
    q3 = median(upper)

    iqr = q3 - q1

    if iqr == 0:
        return [float(x - med) for x in original]

    return [float((x - med) / iqr) for x in original]