def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    # Write code here
    values = values.copy()
    n = len(values)

    i = 0
    while i < n:
        if values[i] is None:
            start = i - 1  # Last known value

            # Find the next known value
            j = i
            while j < n and values[j] is None:
                j += 1

            # Interpolate only if both ends exist
            if start >= 0 and j < n:
                left = values[start]
                right = values[j]
                gap = j - start

                for k in range(1, gap):
                    values[start + k] = left + (right - left) * k / gap

            i = j
        else:
            i += 1

    return values
    