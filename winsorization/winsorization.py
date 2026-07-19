import math
def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    # Write code here
    arr = sorted(values)
    n = len(arr)

    def percentile(p):
        k = (n - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return arr[f]

        return arr[f] + (k - f) * (arr[c] - arr[f])

    lower = percentile(lower_pct)
    upper = percentile(upper_pct)

    result = []
    for x in values:
        if x < lower:
            result.append(lower)
        elif x > upper:
            result.append(upper)
        else:
            result.append(x)

    return result