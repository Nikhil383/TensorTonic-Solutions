def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """

    q = grad[:]
    alpha = []

    # First loop: go backwards through the stored history
    for s, y in reversed(list(zip(s_list, y_list))):
        sy = _dot(s, y)

        if sy <= 0:
            alpha.append(0.0)
            continue

        rho = 1.0 / sy
        a = rho * _dot(s, q)

        alpha.append(a)

        q = [
            qi - a * yi
            for qi, yi in zip(q, y)
        ]

    # Initial Hessian scaling
    if s_list and y_list:
        s = s_list[-1]
        y = y_list[-1]

        yy = _dot(y, y)
        sy = _dot(s, y)

        if yy > 0 and sy > 0:
            gamma = sy / yy
        else:
            gamma = 1.0
    else:
        gamma = 1.0

    # Apply initial inverse-Hessian approximation
    r = [
        gamma * qi
        for qi in q
    ]

    # Second loop: go forward through the history
    for (s, y), a in zip(zip(s_list, y_list), reversed(alpha)):
        sy = _dot(s, y)

        if sy <= 0:
            continue

        rho = 1.0 / sy
        beta = rho * _dot(y, r)

        r = [
            ri + s_i * (a - beta)
            for ri, s_i in zip(r, s)
        ]

    # L-BFGS uses the negative gradient direction
    return [-x for x in r]