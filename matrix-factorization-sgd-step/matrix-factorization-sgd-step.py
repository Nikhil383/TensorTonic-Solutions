def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Perform one SGD step for matrix factorization.
    """
    # Compute dot product
    dot = sum(U[i] * V[i] for i in range(len(U)))

    # Compute prediction error
    error = r - dot

    # Compute updated U using original U and V
    U_new = [
        U[i] + lr * (error * V[i] - reg * U[i])
        for i in range(len(U))
    ]

    # Compute updated V using ORIGINAL U values
    V_new = [
        V[i] + lr * (error * U[i] - reg * V[i])
        for i in range(len(V))
    ]

    return U_new, V_new