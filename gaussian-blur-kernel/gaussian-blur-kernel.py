def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    kernel = []
    center = size // 2
    total = 0

    # Generate kernel values
    for i in range(size):
        row = []
        for j in range(size):
            x = i - center
            y = j - center
            value = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            row.append(value)
            total += value
        kernel.append(row)

    # Normalize the kernel
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total

    return kernel